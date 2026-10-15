/** @file
  Private data and interfaces of the paravirtualized GPU driver.
**/

#ifndef REIMS_VGPU_H_
#define REIMS_VGPU_H_

//
// The device's PCI identity, checked by the driver binding and written into
// the option ROM header. OptionRom.S includes this header for these values
// only, so everything below them is C.
//
#define REIMS_VGPU_VENDOR_ID   0x106B
#define REIMS_VGPU_DEVICE_ID   0xEEEE
#define REIMS_VGPU_CLASS_CODE  0x030200

#ifndef __ASSEMBLER__

#include "Uefi.h"

//
// The EFI display's 32-bit registers in memory BAR 0, named as the device
// names them (docs/testing.md sections 6.2 and 6.22).
//
#define REIMS_VGPU_REG_EFI_ENABLE       0x1200  // nonzero enables the EFI display
#define REIMS_VGPU_REG_EFI_MODE_COUNT   0x1204  // the display's mode count
#define REIMS_VGPU_REG_EFI_MODE_SELECT  0x1208  // selects the mode 0x120C reads
#define REIMS_VGPU_REG_EFI_MODE_VALUE   0x120C  // width | height << 16; all ones out of range
#define REIMS_VGPU_REG_EFI_START        0x1210  // the framebuffer's start, in guest pages (4 KiB on x86-64)
#define REIMS_VGPU_REG_EFI_LENGTH       0x1214  // bytes: whole guest pages, at most 64 MiB
#define REIMS_VGPU_REG_EFI_DEPTH        0x1218  // bits per pixel: only 32 presents
#define REIMS_VGPU_REG_EFI_MODE         0x121C  // width | height << 16

#define REIMS_VGPU_MAX_MODES  127

//
// The memory BAR through which QEMU's apple-gfx-pci shows the frame buffer
// the EFI display's registers name: the Graphics Output Protocol gives its
// address when the buffer reads back through it (docs/architecture.md 11.4).
//
#define REIMS_VGPU_FRAMEBUFFER_BAR  2

typedef struct {
  UINT32    Width;
  UINT32    Height;
  UINT32    BitsPerPixel;
} REIMS_VGPU_MODE;

#define REIMS_VGPU_PRIVATE_DATA_SIGNATURE  SIGNATURE_32 ('R', 'V', 'G', 'P')

typedef struct {
  UINTN                                   Signature;
  EFI_HANDLE                              Handle;
  EFI_PCI_IO_PROTOCOL                     *PciIo;
  UINT64                                  OriginalPciAttributes;
  EFI_DEVICE_PATH_PROTOCOL                *GopDevicePath;
  EFI_GRAPHICS_OUTPUT_PROTOCOL            GraphicsOutput;
  EFI_GRAPHICS_OUTPUT_PROTOCOL_MODE       GraphicsOutputMode;
  EFI_GRAPHICS_OUTPUT_MODE_INFORMATION    ModeInformation;
  UINT64                                  Reserved[4];
  REIMS_VGPU_MODE                         *ModeTable;
  UINT32                                  MaxMode;
  VOID                                    *FrameBuffer;
  EFI_PHYSICAL_ADDRESS                    FrameBufferDeviceAddress;
  VOID                                    *FrameBufferMapping;
  EFI_PHYSICAL_ADDRESS                    FrameBufferBase;  // the Graphics Output Protocol's
} REIMS_VGPU_PRIVATE_DATA;

#define REIMS_VGPU_PRIVATE_DATA_FROM_GRAPHICS_OUTPUT_THIS(a) \
  BASE_CR (a, REIMS_VGPU_PRIVATE_DATA, GraphicsOutput)

extern EFI_HANDLE         gImageHandle;
extern EFI_BOOT_SERVICES  *gBS;

extern EFI_DRIVER_BINDING_PROTOCOL                gReimsVgpuDriverBinding;
extern EFI_COMPONENT_NAME_PROTOCOL                gReimsVgpuComponentName;
extern EFI_COMPONENT_NAME2_PROTOCOL               gReimsVgpuComponentName2;
extern EFI_DRIVER_SUPPORTED_EFI_VERSION_PROTOCOL  gReimsVgpuSupportedEfiVersion;

EFI_STATUS
EFIAPI
ReimsVgpuComponentNameGetDriverName (
  IN  EFI_COMPONENT_NAME_PROTOCOL  *This,
  IN  CHAR8                        *Language,
  OUT CHAR16                       **DriverName
  );

EFI_STATUS
EFIAPI
ReimsVgpuComponentNameGetControllerName (
  IN  EFI_COMPONENT_NAME_PROTOCOL  *This,
  IN  EFI_HANDLE                   ControllerHandle,
  IN  EFI_HANDLE                   ChildHandle OPTIONAL,
  IN  CHAR8                        *Language,
  OUT CHAR16                       **ControllerName
  );

VOID *
AllocateRuntimePool (
  IN UINTN  AllocationSize
  );

VOID
ReimsVgpuWriteRegister (
  IN EFI_PCI_IO_PROTOCOL  *PciIo,
  IN UINT64               Offset,
  IN UINT32               Value
  );

VOID
ReimsVgpuGraphicsOutputConstructor (
  IN REIMS_VGPU_PRIVATE_DATA  *Private
  );

#endif // __ASSEMBLER__

#endif
