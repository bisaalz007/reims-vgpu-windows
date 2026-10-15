/** @file
  Graphics Output Protocol on a linear 32-bit frame buffer.
**/

#include "ReimsVgpu.h"

//
// A rectangle copy between two surfaces, each described by its row pitch in
// bytes and the pixel coordinates of the rectangle's first corner.
//
typedef struct {
  UINT32    SourcePitch;
  UINT32    SourceX;
  UINT32    SourceY;
  UINT32    Width;
  UINT32    Height;
  UINT32    DestinationPitch;
  UINT32    DestinationX;
  UINT32    DestinationY;
} REIMS_VGPU_BLT_RECT;

STATIC
VOID
ReimsVgpuSetModeInformation (
  OUT EFI_GRAPHICS_OUTPUT_MODE_INFORMATION  *Info
  )
{
  Info->Version                       = 0;
  Info->PixelFormat                   = PixelBlueGreenRedReserved8BitPerColor;
  Info->PixelInformation.RedMask      = 0xFF000000;
  Info->PixelInformation.GreenMask    = 0x00FF0000;
  Info->PixelInformation.BlueMask     = 0x0000FF00;
  Info->PixelInformation.ReservedMask = 0x000000FF;
  Info->PixelsPerScanLine             = Info->HorizontalResolution;
}

/**
  Publish the frame buffer for the mode just stored in Mode->Info.
**/
STATIC
VOID
ReimsVgpuUpdateMode (
  IN REIMS_VGPU_PRIVATE_DATA            *Private,
  IN EFI_GRAPHICS_OUTPUT_PROTOCOL_MODE  *Mode
  )
{
  EFI_GRAPHICS_OUTPUT_MODE_INFORMATION  *Info;

  Info = Mode->Info;
  ReimsVgpuSetModeInformation (Info);
  Mode->FrameBufferBase = Private->FrameBufferBase;
  Mode->FrameBufferSize = Info->HorizontalResolution * Info->VerticalResolution * 4;
}

EFI_STATUS
EFIAPI
ReimsVgpuGraphicsOutputQueryMode (
  IN  EFI_GRAPHICS_OUTPUT_PROTOCOL          *This,
  IN  UINT32                                ModeNumber,
  OUT UINTN                                 *SizeOfInfo,
  OUT EFI_GRAPHICS_OUTPUT_MODE_INFORMATION  **Info
  )
{
  REIMS_VGPU_PRIVATE_DATA  *Private;

  Private = REIMS_VGPU_PRIVATE_DATA_FROM_GRAPHICS_OUTPUT_THIS (This);

  if ((Info == NULL) || (SizeOfInfo == NULL) || (ModeNumber >= This->Mode->MaxMode)) {
    return EFI_INVALID_PARAMETER;
  }

  *Info = AllocateRuntimePool (sizeof (EFI_GRAPHICS_OUTPUT_MODE_INFORMATION));
  if (*Info == NULL) {
    return EFI_OUT_OF_RESOURCES;
  }

  *SizeOfInfo = sizeof (EFI_GRAPHICS_OUTPUT_MODE_INFORMATION);

  (*Info)->HorizontalResolution = Private->ModeTable[ModeNumber].Width;
  (*Info)->VerticalResolution   = Private->ModeTable[ModeNumber].Height;
  ReimsVgpuSetModeInformation (*Info);

  return EFI_SUCCESS;
}

EFI_STATUS
EFIAPI
ReimsVgpuGraphicsOutputSetMode (
  IN  EFI_GRAPHICS_OUTPUT_PROTOCOL  *This,
  IN  UINT32                        ModeNumber
  )
{
  REIMS_VGPU_PRIVATE_DATA  *Private;

  Private = REIMS_VGPU_PRIVATE_DATA_FROM_GRAPHICS_OUTPUT_THIS (This);

  if (ModeNumber >= This->Mode->MaxMode) {
    return EFI_UNSUPPORTED;
  }

  This->Mode->Mode                       = ModeNumber;
  This->Mode->Info->HorizontalResolution = Private->ModeTable[ModeNumber].Width;
  This->Mode->Info->VerticalResolution   = Private->ModeTable[ModeNumber].Height;
  This->Mode->SizeOfInfo                 = sizeof (EFI_GRAPHICS_OUTPUT_MODE_INFORMATION);
  ReimsVgpuUpdateMode (Private, This->Mode);

  ReimsVgpuWriteRegister (Private->PciIo, REIMS_VGPU_REG_EFI_DEPTH, Private->ModeTable[ModeNumber].BitsPerPixel);
  ReimsVgpuWriteRegister (
    Private->PciIo,
    REIMS_VGPU_REG_EFI_MODE,
    (Private->ModeTable[ModeNumber].Width & 0xFFFF) | (Private->ModeTable[ModeNumber].Height << 16)
    );

  return EFI_SUCCESS;
}

/**
  Copy a rectangle of 32-bit pixels between two surfaces.
**/
STATIC
VOID
ReimsVgpuCopyRect (
  IN  UINT8                *Source,
  OUT UINT8                *Destination,
  IN  REIMS_VGPU_BLT_RECT  Rect
  )
{
  UINT32  X;
  UINT32  Y;
  UINT32  SourceX;
  UINT32  SourceY;
  UINT32  DestinationX;
  UINT32  DestinationY;
  UINT8   *SourceRow;
  UINT8   *DestinationRow;

  for (Y = 0; Y < Rect.Height; Y++) {
    SourceY        = Rect.SourceY + Y;
    DestinationY   = Rect.DestinationY + Y;
    SourceRow      = Source + SourceY * Rect.SourcePitch;
    DestinationRow = Destination + DestinationY * Rect.DestinationPitch;
    for (X = 0; X < Rect.Width; X++) {
      SourceX      = Rect.SourceX + X;
      DestinationX = Rect.DestinationX + X;
      *(UINT32 *)(DestinationRow + DestinationX * sizeof (UINT32)) =
        *(UINT32 *)(SourceRow + SourceX * sizeof (UINT32));
    }
  }
}

/**
  Copy a rectangle within one surface. Rows and columns are walked backward
  when the destination lies after the source.
**/
STATIC
VOID
ReimsVgpuMoveRect (
  IN OUT UINT8                *Buffer,
  IN     REIMS_VGPU_BLT_RECT  Rect
  )
{
  UINT32  StartDestinationX;
  UINT32  StartSourceX;
  UINT32  StartSourceY;
  UINT32  StartDestinationY;
  UINT32  X;
  UINT32  Y;
  UINT32  OffsetX;
  UINT32  OffsetY;
  UINT32  SourceX;
  UINT32  SourceY;
  UINT32  DestinationX;
  UINT32  DestinationY;
  UINT32  *SourceRow;
  UINT32  *DestinationRow;

  if (Rect.SourceX < Rect.DestinationX) {
    StartSourceX      = Rect.SourceX + Rect.Width;
    StartDestinationX = StartSourceX + Rect.Width;
  } else {
    StartSourceX      = Rect.SourceX;
    StartDestinationX = Rect.DestinationX;
  }

  if (Rect.SourceY < Rect.DestinationY) {
    StartSourceY      = Rect.SourceY + Rect.Height;
    StartDestinationY = StartSourceY + Rect.Height;
  } else {
    StartSourceY      = Rect.SourceY;
    StartDestinationY = Rect.DestinationY;
  }

  for (Y = 0; Y < Rect.Height; Y++) {
    OffsetY        = (Rect.SourceY < Rect.DestinationY) ? -Y : Y;
    SourceY        = StartSourceY + OffsetY;
    DestinationY   = StartDestinationY + OffsetY;
    SourceRow      = (UINT32 *)((UINTN)Buffer + SourceY * Rect.SourcePitch);
    DestinationRow = (UINT32 *)((UINTN)Buffer + DestinationY * Rect.DestinationPitch);
    for (X = 0; X < Rect.Width; X++) {
      OffsetX                      = (Rect.SourceX < Rect.DestinationX) ? -X : X;
      SourceX                      = StartSourceX + OffsetX;
      DestinationX                 = StartDestinationX + OffsetX;
      DestinationRow[DestinationX] = SourceRow[SourceX];
    }
  }
}

EFI_STATUS
EFIAPI
ReimsVgpuGraphicsOutputBlt (
  IN  EFI_GRAPHICS_OUTPUT_PROTOCOL       *This,
  IN  EFI_GRAPHICS_OUTPUT_BLT_PIXEL      *BltBuffer   OPTIONAL,
  IN  EFI_GRAPHICS_OUTPUT_BLT_OPERATION  BltOperation,
  IN  UINTN                              SourceX,
  IN  UINTN                              SourceY,
  IN  UINTN                              DestinationX,
  IN  UINTN                              DestinationY,
  IN  UINTN                              Width,
  IN  UINTN                              Height,
  IN  UINTN                              Delta         OPTIONAL
  )
{
  REIMS_VGPU_PRIVATE_DATA  *Private;
  REIMS_VGPU_BLT_RECT      Rect;
  UINT32                   X;
  UINT32                   Y;
  EFI_GRAPHICS_OUTPUT_BLT_PIXEL  *Row;

  Private = REIMS_VGPU_PRIVATE_DATA_FROM_GRAPHICS_OUTPUT_THIS (This);

  switch (BltOperation) {
    case EfiBltVideoFill:
      Y = MAX_UINT32;
      while (++Y < Height) {
        Row = (EFI_GRAPHICS_OUTPUT_BLT_PIXEL *)((UINT8 *)Private->FrameBuffer +
                                                (UINT32)(DestinationY + Y) * This->Mode->Info->HorizontalResolution * 4);
        for (X = 0; X < Width; X++) {
          Row[(UINT32)(DestinationX + X)] = *BltBuffer;
        }
      }

      break;

    case EfiBltVideoToBltBuffer:
      Rect.SourcePitch      = This->Mode->Info->HorizontalResolution * 4;
      Rect.SourceX          = (UINT32)SourceX;
      Rect.SourceY          = (UINT32)SourceY;
      Rect.Width            = (UINT32)Width;
      Rect.Height           = (UINT32)Height;
      Rect.DestinationPitch = (UINT32)((Delta != 0) ? Delta : Width * sizeof (EFI_GRAPHICS_OUTPUT_BLT_PIXEL));
      Rect.DestinationX     = (UINT32)DestinationX;
      Rect.DestinationY     = (UINT32)DestinationY;
      ReimsVgpuCopyRect (Private->FrameBuffer, (UINT8 *)BltBuffer, Rect);
      break;

    case EfiBltBufferToVideo:
      Rect.SourcePitch      = (UINT32)((Delta != 0) ? Delta : Width * sizeof (EFI_GRAPHICS_OUTPUT_BLT_PIXEL));
      Rect.SourceX          = (UINT32)SourceX;
      Rect.SourceY          = (UINT32)SourceY;
      Rect.Width            = (UINT32)Width;
      Rect.Height           = (UINT32)Height;
      Rect.DestinationPitch = This->Mode->Info->HorizontalResolution * 4;
      Rect.DestinationX     = (UINT32)DestinationX;
      Rect.DestinationY     = (UINT32)DestinationY;
      ReimsVgpuCopyRect ((UINT8 *)BltBuffer, Private->FrameBuffer, Rect);
      break;

    case EfiBltVideoToVideo:
      Rect.SourcePitch      = This->Mode->Info->HorizontalResolution * 4;
      Rect.SourceX          = (UINT32)SourceX;
      Rect.SourceY          = (UINT32)SourceY;
      Rect.Width            = (UINT32)Width;
      Rect.Height           = (UINT32)Height;
      Rect.DestinationPitch = This->Mode->Info->HorizontalResolution * 4;
      Rect.DestinationX     = (UINT32)DestinationX;
      Rect.DestinationY     = (UINT32)DestinationY;
      ReimsVgpuMoveRect (Private->FrameBuffer, Rect);
      break;

    default:
      return EFI_INVALID_PARAMETER;
  }

  return EFI_SUCCESS;
}

VOID
ReimsVgpuGraphicsOutputConstructor (
  IN REIMS_VGPU_PRIVATE_DATA  *Private
  )
{
  EFI_GRAPHICS_OUTPUT_PROTOCOL  *GraphicsOutput;

  GraphicsOutput            = &Private->GraphicsOutput;
  GraphicsOutput->QueryMode = ReimsVgpuGraphicsOutputQueryMode;
  GraphicsOutput->SetMode   = ReimsVgpuGraphicsOutputSetMode;
  GraphicsOutput->Blt       = ReimsVgpuGraphicsOutputBlt;
  GraphicsOutput->Mode      = &Private->GraphicsOutputMode;

  GraphicsOutput->Mode->MaxMode    = Private->MaxMode;
  GraphicsOutput->Mode->Mode       = 0;
  GraphicsOutput->Mode->Info       = &Private->ModeInformation;
  GraphicsOutput->Mode->SizeOfInfo = sizeof (EFI_GRAPHICS_OUTPUT_MODE_INFORMATION);

  ReimsVgpuGraphicsOutputSetMode (GraphicsOutput, 0);
}
