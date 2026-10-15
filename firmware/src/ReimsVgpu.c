/** @file
  Entry point and driver binding for the paravirtualized GPU: device
  detection, mode discovery, frame buffer setup, and Graphics Output
  installation.
**/

#include "ReimsVgpu.h"

EFI_HANDLE         gImageHandle;
EFI_BOOT_SERVICES  *gBS;

STATIC EFI_DEVICE_PATH_UTILITIES_PROTOCOL  *mDevicePathUtilities;

EFI_GUID  gEfiDevicePathProtocolGuid                = EFI_DEVICE_PATH_PROTOCOL_GUID;
EFI_GUID  gEfiDevicePathUtilitiesProtocolGuid       = EFI_DEVICE_PATH_UTILITIES_PROTOCOL_GUID;
EFI_GUID  gEfiDriverBindingProtocolGuid             = EFI_DRIVER_BINDING_PROTOCOL_GUID;
EFI_GUID  gEfiComponentNameProtocolGuid             = EFI_COMPONENT_NAME_PROTOCOL_GUID;
EFI_GUID  gEfiComponentName2ProtocolGuid            = EFI_COMPONENT_NAME2_PROTOCOL_GUID;
EFI_GUID  gEfiDriverSupportedEfiVersionProtocolGuid = EFI_DRIVER_SUPPORTED_EFI_VERSION_PROTOCOL_GUID;
EFI_GUID  gEfiPciIoProtocolGuid                     = EFI_PCI_IO_PROTOCOL_GUID;
EFI_GUID  gEfiGraphicsOutputProtocolGuid            = EFI_GRAPHICS_OUTPUT_PROTOCOL_GUID;

//
// Modes offered when the device does not report a usable mode list: one
// each for 16:10, 16:9, and 4:3.
//
STATIC CONST REIMS_VGPU_MODE  mReimsVgpuDefaultModes[] = {
  { 1920, 1200, 32 },
  { 1920, 1080, 32 },
  { 1600, 1200, 32 }
};

EFI_DRIVER_SUPPORTED_EFI_VERSION_PROTOCOL  gReimsVgpuSupportedEfiVersion = {
  sizeof (EFI_DRIVER_SUPPORTED_EFI_VERSION_PROTOCOL),
  0
};

/**
  Allocate pool memory of MemoryType, or return NULL on failure.
**/
STATIC
VOID *
InternalAllocatePool (
  IN EFI_MEMORY_TYPE  MemoryType,
  IN UINTN            AllocationSize
  )
{
  EFI_STATUS  Status;
  VOID        *Memory;

  Status = gBS->AllocatePool (MemoryType, AllocationSize, &Memory);
  if (EFI_ERROR (Status)) {
    Memory = NULL;
  }

  return Memory;
}

VOID *
AllocateRuntimePool (
  IN UINTN  AllocationSize
  )
{
  return InternalAllocatePool (EfiRuntimeServicesData, AllocationSize);
}

STATIC
VOID *
AllocateZeroPool (
  IN UINTN  AllocationSize
  )
{
  UINT8  *Memory;
  UINTN  Index;

  Memory = InternalAllocatePool (EfiBootServicesData, AllocationSize);
  if (Memory != NULL) {
    for (Index = 0; Index < AllocationSize; Index++) {
      Memory[Index] = 0;
    }
  }

  return Memory;
}

STATIC
VOID
FreePool (
  IN VOID  *Buffer
  )
{
  gBS->FreePool (Buffer);
}

/**
  Check whether Controller is the paravirtualized GPU.
**/
EFI_STATUS
EFIAPI
ReimsVgpuDriverBindingSupported (
  IN EFI_DRIVER_BINDING_PROTOCOL  *This,
  IN EFI_HANDLE                   Controller,
  IN EFI_DEVICE_PATH_PROTOCOL     *RemainingDevicePath
  )
{
  EFI_STATUS           Status;
  EFI_PCI_IO_PROTOCOL  *PciIo;
  PCI_TYPE00           Pci;

  Status = gBS->OpenProtocol (
                  Controller,
                  &gEfiPciIoProtocolGuid,
                  (VOID **)&PciIo,
                  This->DriverBindingHandle,
                  Controller,
                  EFI_OPEN_PROTOCOL_BY_DRIVER
                  );
  if (EFI_ERROR (Status)) {
    return Status;
  }

  Status = PciIo->Pci.Read (
                        PciIo,
                        EfiPciIoWidthUint32,
                        0,
                        sizeof (Pci) / sizeof (UINT32),
                        &Pci
                        );
  if (!EFI_ERROR (Status)) {
    if ((Pci.Hdr.VendorId == REIMS_VGPU_VENDOR_ID) &&
        (Pci.Hdr.DeviceId == REIMS_VGPU_DEVICE_ID))
    {
      Status = EFI_SUCCESS;
    } else {
      Status = EFI_UNSUPPORTED;
    }
  }

  gBS->CloseProtocol (
         Controller,
         &gEfiPciIoProtocolGuid,
         This->DriverBindingHandle,
         Controller
         );

  return Status;
}

/**
  Bind to Controller: read the device's mode list, allocate and map a frame
  buffer large enough for the largest mode, and install Device Path and
  Graphics Output protocols on a child handle.
**/
/**
  Write a 32-bit device register in BAR 0.
**/
VOID
ReimsVgpuWriteRegister (
  IN EFI_PCI_IO_PROTOCOL  *PciIo,
  IN UINT64               Offset,
  IN UINT32               Value
  )
{
  PciIo->Mem.Write (PciIo, EfiPciIoWidthUint32, 0, Offset, 1, &Value);
}

/**
  Read a 32-bit device register in BAR 0.
**/
STATIC
VOID
ReimsVgpuReadRegister (
  IN  EFI_PCI_IO_PROTOCOL  *PciIo,
  IN  UINT64               Offset,
  OUT UINT32               *Value
  )
{
  PciIo->Mem.Read (PciIo, EfiPciIoWidthUint32, 0, Offset, 1, Value);
}

/**
  Check that the frame buffer's dword at Index reads back through the frame
  buffer BAR: a pattern written into the buffer, then its complement. The
  dword is restored.
**/
STATIC
BOOLEAN
ReimsVgpuFrameBufferReadsBack (
  IN REIMS_VGPU_PRIVATE_DATA  *Private,
  IN UINT32                   Index
  )
{
  STATIC CONST UINT32  Patterns[] = { 0x5AA5C33C, 0xA55A3CC3 };
  volatile UINT32      *Buffer;
  UINT32               Original;
  UINT32               Value;
  UINTN                Pattern;
  BOOLEAN              ReadsBack;
  EFI_STATUS           Status;

  Buffer    = (volatile UINT32 *)Private->FrameBuffer;
  Original  = Buffer[Index];
  ReadsBack = TRUE;
  for (Pattern = 0; Pattern < ARRAY_SIZE (Patterns) && ReadsBack; Pattern++) {
    Buffer[Index] = Patterns[Pattern];
    //
    // The store is visible before the read through the BAR, another
    // address of the same memory.
    //
    __asm__ __volatile__ ("mfence" ::: "memory");
    Status = Private->PciIo->Mem.Read (
                               Private->PciIo,
                               EfiPciIoWidthUint32,
                               REIMS_VGPU_FRAMEBUFFER_BAR,
                               (UINT64)Index * sizeof (UINT32),
                               1,
                               &Value
                               );
    ReadsBack = !EFI_ERROR (Status) && (Value == Patterns[Pattern]);
  }

  Buffer[Index] = Original;
  return ReadsBack;
}

/**
  The frame buffer's address for the Graphics Output Protocol: the frame
  buffer BAR's, where the device shows the Size bytes of the frame buffer
  there, or else the frame buffer's own in guest RAM.

  The BAR is used when it is a memory BAR of at least Size bytes, below
  4 GiB with no address translation, and the buffer's first and last dwords
  read back through it. QEMU's apple-gfx-pci makes BAR 2 an alias of the RAM
  the EFI display's registers name, once they are written.
**/
STATIC
EFI_PHYSICAL_ADDRESS
ReimsVgpuFrameBufferBase (
  IN REIMS_VGPU_PRIVATE_DATA  *Private,
  IN UINT32                   Size
  )
{
  EFI_STATUS                         Status;
  EFI_ACPI_ADDRESS_SPACE_DESCRIPTOR  *Bar;
  EFI_PHYSICAL_ADDRESS               BarBase;
  BOOLEAN                            Usable;

  Status = Private->PciIo->GetBarAttributes (
                             Private->PciIo,
                             REIMS_VGPU_FRAMEBUFFER_BAR,
                             NULL,
                             (VOID **)&Bar
                             );
  if (EFI_ERROR (Status)) {
    return Private->FrameBufferDeviceAddress;
  }

  BarBase = Bar->AddrRangeMin;
  Usable  = (Bar->Desc == ACPI_ADDRESS_SPACE_DESCRIPTOR) &&
            (Bar->ResType == ACPI_ADDRESS_SPACE_TYPE_MEM) &&
            (Bar->AddrTranslationOffset == 0) &&
            (Bar->AddrLen >= Size) &&
            (BarBase < SIZE_4GB) &&
            (Size <= SIZE_4GB - BarBase);
  FreePool (Bar);

  if (!Usable ||
      !ReimsVgpuFrameBufferReadsBack (Private, 0) ||
      !ReimsVgpuFrameBufferReadsBack (Private, Size / sizeof (UINT32) - 1))
  {
    return Private->FrameBufferDeviceAddress;
  }

  return BarBase;
}

EFI_STATUS
EFIAPI
ReimsVgpuDriverBindingStart (
  IN EFI_DRIVER_BINDING_PROTOCOL  *This,
  IN EFI_HANDLE                   Controller,
  IN EFI_DEVICE_PATH_PROTOCOL     *RemainingDevicePath
  )
{
  EFI_STATUS                Status;
  REIMS_VGPU_PRIVATE_DATA   *Private;
  EFI_DEVICE_PATH_PROTOCOL  *ParentDevicePath;
  ACPI_ADR_DEVICE_PATH      AcpiDeviceNode;
  UINTN                     Index;
  UINT32                    Size;
  UINT32                    MaxSize;
  UINTN                     MapSize;

  Private = AllocateZeroPool (sizeof (REIMS_VGPU_PRIVATE_DATA));
  if (Private == NULL) {
    return EFI_OUT_OF_RESOURCES;
  }

  Private->Signature = REIMS_VGPU_PRIVATE_DATA_SIGNATURE;
  Private->Handle    = NULL;

  Status = gBS->OpenProtocol (
                  Controller,
                  &gEfiPciIoProtocolGuid,
                  (VOID **)&Private->PciIo,
                  This->DriverBindingHandle,
                  Controller,
                  EFI_OPEN_PROTOCOL_BY_DRIVER
                  );
  if (EFI_ERROR (Status)) {
    return Status;
  }

  Status = Private->PciIo->Attributes (
                             Private->PciIo,
                             EfiPciIoAttributeOperationGet,
                             0,
                             &Private->OriginalPciAttributes
                             );
  if (EFI_ERROR (Status)) {
    return Status;
  }

  Status = Private->PciIo->Attributes (
                             Private->PciIo,
                             EfiPciIoAttributeOperationEnable,
                             EFI_PCI_DEVICE_ENABLE,
                             NULL
                             );
  if (EFI_ERROR (Status)) {
    return Status;
  }

  Status = gBS->HandleProtocol (
                  Controller,
                  &gEfiDevicePathProtocolGuid,
                  (VOID **)&ParentDevicePath
                  );
  if (EFI_ERROR (Status)) {
    return Status;
  }

  UINT32  ModeCount;

  //
  // A resolution of all ones means the list changed while it was being
  // read, so start over.
  //
  ReimsVgpuWriteRegister (Private->PciIo, REIMS_VGPU_REG_EFI_ENABLE, 1);

Retry:
  if (Private->ModeTable != NULL) {
    FreePool (Private->ModeTable);
    Private->ModeTable = NULL;
    Private->MaxMode   = 0;
  }

  ReimsVgpuReadRegister (Private->PciIo, REIMS_VGPU_REG_EFI_MODE_COUNT, &ModeCount);
  if ((ModeCount > 0) && (ModeCount <= REIMS_VGPU_MAX_MODES)) {
    Private->ModeTable = AllocateZeroPool (ModeCount * sizeof (REIMS_VGPU_MODE));
    if (Private->ModeTable == NULL) {
      return EFI_OUT_OF_RESOURCES;
    }

    Private->MaxMode = ModeCount;
    for (Index = 0; Index < ModeCount; Index++) {
      UINT32  Resolution;
      UINT32  Width;
      UINT32  Height;

      ReimsVgpuWriteRegister (Private->PciIo, REIMS_VGPU_REG_EFI_MODE_SELECT, (UINT32)Index);
      ReimsVgpuReadRegister (Private->PciIo, REIMS_VGPU_REG_EFI_MODE_VALUE, &Resolution);
      if (Resolution == 0xFFFFFFFF) {
        goto Retry;
      }

      Height = Resolution >> 16;
      Width  = Resolution & 0xFFFF;
      Private->ModeTable[Index].Width        = Width;
      Private->ModeTable[Index].Height       = Height;
      Private->ModeTable[Index].BitsPerPixel = 32;
    }
  } else {
    Private->ModeTable = AllocateZeroPool (sizeof (mReimsVgpuDefaultModes));
    if (Private->ModeTable == NULL) {
      return EFI_OUT_OF_RESOURCES;
    }

    Private->MaxMode = ARRAY_SIZE (mReimsVgpuDefaultModes);
    for (Index = 0; Index < Private->MaxMode; Index++) {
      Private->ModeTable[Index] = mReimsVgpuDefaultModes[Index];
    }
  }

  //
  // Size the frame buffer for the largest mode.
  //
  MaxSize = 0;
  for (Index = 0; Index < Private->MaxMode; Index++) {
    Size = Private->ModeTable[Index].Width * Private->ModeTable[Index].Height *
           (Private->ModeTable[Index].BitsPerPixel / 8);
    MaxSize = MAX (Size, MaxSize);
  }

  Size  = (MaxSize + EFI_PAGE_MASK) & ~EFI_PAGE_MASK;

  //
  // Runtime services data, which the OS keeps past ExitBootServices: the
  // device presents these pages, and the frame buffer BAR shows them. macOS's
  // boot.efi may move runtime data, which would leave both on the old pages,
  // unless the loader keeps it in place, as OpenCore's AvoidRuntimeDefrag
  // quirk does.
  //
  Status = Private->PciIo->AllocateBuffer (
                             Private->PciIo,
                             AllocateAnyPages,
                             EfiRuntimeServicesData,
                             EFI_SIZE_TO_PAGES (Size),
                             &Private->FrameBuffer,
                             0
                             );
  if (EFI_ERROR (Status)) {
    return Status;
  }

  MapSize = Size;
  Status  = Private->PciIo->Map (
                              Private->PciIo,
                              EfiPciIoOperationBusMasterCommonBuffer,
                              Private->FrameBuffer,
                              &MapSize,
                              &Private->FrameBufferDeviceAddress,
                              &Private->FrameBufferMapping
                              );
  if (!EFI_ERROR (Status) && (MapSize != Size)) {
    Private->PciIo->Unmap (Private->PciIo, Private->FrameBufferMapping);
    Status = EFI_OUT_OF_RESOURCES;
  }

  if (EFI_ERROR (Status)) {
    Private->PciIo->FreeBuffer (Private->PciIo, EFI_SIZE_TO_PAGES (Size), &Private->FrameBuffer);
    return Status;
  }

  //
  // These are in 4 KiB pages, the x86-64 guest's, which the device counts when
  // the library is built for x86-64 (docs/architecture.md 11.4).
  //
  ReimsVgpuWriteRegister (Private->PciIo, REIMS_VGPU_REG_EFI_START, (UINT32)(Private->FrameBufferDeviceAddress >> EFI_PAGE_SHIFT));
  ReimsVgpuWriteRegister (Private->PciIo, REIMS_VGPU_REG_EFI_LENGTH, Size);

  //
  // The device keeps presenting the buffer in RAM; the Graphics Output
  // Protocol gives the frame buffer BAR's address where the BAR shows it. An
  // OS with no driver of the device's own (macOS's installers and recovery)
  // takes the boot frame buffer over only inside one of the device's BARs.
  //
  Private->FrameBufferBase = ReimsVgpuFrameBufferBase (Private, Size);

  //
  // The Graphics Output child is an ACPI _ADR node below the controller.
  //
  AcpiDeviceNode.Header.Type    = ACPI_DEVICE_PATH;
  AcpiDeviceNode.Header.SubType = ACPI_ADR_DP;
  AcpiDeviceNode.ADR            = ACPI_DISPLAY_ADR (1, 0, 0, 1, 0, ACPI_ADR_DISPLAY_TYPE_OTHER, 0, 0);
  AcpiDeviceNode.Header.Length[0] = (UINT8)sizeof (ACPI_ADR_DEVICE_PATH);
  AcpiDeviceNode.Header.Length[1] = 0;

  Private->GopDevicePath = mDevicePathUtilities->AppendDeviceNode (ParentDevicePath, &AcpiDeviceNode.Header);

  Private->Handle = NULL;
  Status          = gBS->InstallMultipleProtocolInterfaces (
                           &Private->Handle,
                           &gEfiDevicePathProtocolGuid,
                           Private->GopDevicePath,
                           NULL
                           );
  if (EFI_ERROR (Status)) {
    return Status;
  }

  Status = gBS->OpenProtocol (
                  Controller,
                  &gEfiPciIoProtocolGuid,
                  (VOID **)&Private->PciIo,
                  gImageHandle,
                  Private->Handle,
                  EFI_OPEN_PROTOCOL_BY_CHILD_CONTROLLER
                  );
  if (EFI_ERROR (Status)) {
    return Status;
  }

  ReimsVgpuGraphicsOutputConstructor (Private);

  Status = gBS->InstallMultipleProtocolInterfaces (
                  &Private->Handle,
                  &gEfiGraphicsOutputProtocolGuid,
                  &Private->GraphicsOutput,
                  NULL
                  );
  return Status;
}

/**
  Remove the Graphics Output child and release its resources.
**/
EFI_STATUS
EFIAPI
ReimsVgpuDriverBindingStop (
  IN EFI_DRIVER_BINDING_PROTOCOL  *This,
  IN EFI_HANDLE                   Controller,
  IN UINTN                        NumberOfChildren,
  IN EFI_HANDLE                   *ChildHandleBuffer
  )
{
  EFI_STATUS                    Status;
  EFI_GRAPHICS_OUTPUT_PROTOCOL  *GraphicsOutput;
  REIMS_VGPU_PRIVATE_DATA       *Private;

  Status = gBS->OpenProtocol (
                  ChildHandleBuffer[0],
                  &gEfiPciIoProtocolGuid,
                  (VOID **)&GraphicsOutput,
                  gImageHandle,
                  Controller,
                  EFI_OPEN_PROTOCOL_GET_PROTOCOL
                  );
  Private = REIMS_VGPU_PRIVATE_DATA_FROM_GRAPHICS_OUTPUT_THIS (GraphicsOutput);
  if (!EFI_ERROR (Status)) {
    return Status;
  }

  if (NumberOfChildren == 0) {
    gBS->CloseProtocol (
           Controller,
           &gEfiPciIoProtocolGuid,
           gImageHandle,
           Controller
           );
    return EFI_SUCCESS;
  }

  gBS->UninstallMultipleProtocolInterfaces (
         &Private->Handle,
         &gEfiDevicePathProtocolGuid,
         Private->GopDevicePath,
         &gEfiGraphicsOutputProtocolGuid,
         &Private->GraphicsOutput,
         NULL
         );

  gBS->CloseProtocol (
         Controller,
         &gEfiPciIoProtocolGuid,
         gImageHandle,
         Private->Handle
         );

  FreePool (Private->GopDevicePath);
  FreePool (Private);

  return EFI_SUCCESS;
}

EFI_DRIVER_BINDING_PROTOCOL  gReimsVgpuDriverBinding = {
  ReimsVgpuDriverBindingSupported,
  ReimsVgpuDriverBindingStart,
  ReimsVgpuDriverBindingStop,
  0x10,
  NULL,
  NULL
};

/**
  Image entry point. Locate the Device Path Utilities protocol, then install
  the driver's protocols on its image handle. Only the status of the last
  install is returned; the lookup and the first install are not checked.
**/
EFI_STATUS
EFIAPI
ReimsVgpuEntryPoint (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  EFI_STATUS  Status;

  gImageHandle = ImageHandle;
  gBS          = SystemTable->BootServices;

  gBS->LocateProtocol (
         &gEfiDevicePathUtilitiesProtocolGuid,
         NULL,
         (VOID **)&mDevicePathUtilities
         );

  gReimsVgpuDriverBinding.ImageHandle         = ImageHandle;
  gReimsVgpuDriverBinding.DriverBindingHandle = ImageHandle;
  gBS->InstallMultipleProtocolInterfaces (
         &gReimsVgpuDriverBinding.DriverBindingHandle,
         &gEfiDriverBindingProtocolGuid,
         &gReimsVgpuDriverBinding,
         &gEfiComponentNameProtocolGuid,
         &gReimsVgpuComponentName,
         &gEfiComponentName2ProtocolGuid,
         &gReimsVgpuComponentName2,
         NULL
         );

  gReimsVgpuSupportedEfiVersion.FirmwareVersion = EFI_2_10_SYSTEM_TABLE_REVISION;
  Status                                        = gBS->InstallMultipleProtocolInterfaces (
                                                         &ImageHandle,
                                                         &gEfiDriverSupportedEfiVersionProtocolGuid,
                                                         &gReimsVgpuSupportedEfiVersion,
                                                         NULL
                                                         );
  return Status;
}
