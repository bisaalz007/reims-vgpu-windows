/** @file
  Component Name and Component Name 2 protocols.
**/

#include "ReimsVgpu.h"

//
// A name and the semicolon-separated languages it is offered in.
//
typedef struct {
  CHAR8     *Language;
  CHAR16    *UnicodeString;
} EFI_UNICODE_STRING_TABLE;

EFI_COMPONENT_NAME_PROTOCOL  gReimsVgpuComponentName = {
  ReimsVgpuComponentNameGetDriverName,
  ReimsVgpuComponentNameGetControllerName,
  "eng"
};

EFI_COMPONENT_NAME2_PROTOCOL  gReimsVgpuComponentName2 = {
  (EFI_COMPONENT_NAME2_GET_DRIVER_NAME)ReimsVgpuComponentNameGetDriverName,
  (EFI_COMPONENT_NAME2_GET_CONTROLLER_NAME)ReimsVgpuComponentNameGetControllerName,
  "en"
};

EFI_UNICODE_STRING_TABLE  mReimsVgpuDriverNameTable[] = {
  { "eng;en", L"Reims vGPU Driver" },
  { NULL,     NULL                 }
};

EFI_UNICODE_STRING_TABLE  mReimsVgpuControllerNameTable[] = {
  { "eng;en", L"Reims vGPU" },
  { NULL,     NULL          }
};

/**
  Compare up to Length characters of two ASCII strings.

  @return  The difference between the first mismatched characters, or zero
           if the strings match up to Length characters.
**/
STATIC
INTN
AsciiStrnCmp (
  IN CONST CHAR8  *FirstString,
  IN CONST CHAR8  *SecondString,
  IN UINTN        Length
  )
{
  if (Length == 0) {
    return 0;
  }

  while ((*FirstString != '\0') &&
         (*SecondString != '\0') &&
         (*FirstString == *SecondString) &&
         (Length > 1))
  {
    FirstString++;
    SecondString++;
    Length--;
  }

  return *FirstString - *SecondString;
}

/**
  Compare two three-character ISO 639-2 language codes.
**/
STATIC
BOOLEAN
CompareIso639LanguageCode (
  IN CONST CHAR8  *Language1,
  IN CONST CHAR8  *Language2
  )
{
  return (BOOLEAN)((Language1[0] == Language2[0]) &&
                   (Language1[1] == Language2[1]) &&
                   (Language1[2] == Language2[2]));
}

/**
  Test whether TargetLanguage appears in a semicolon-separated list of
  RFC 4646 language codes.
**/
STATIC
EFI_STATUS
IsLanguageSupported (
  IN CONST CHAR8  *SupportedLanguages,
  IN CONST CHAR8  *TargetLanguage
  )
{
  UINTN  Index;

  while (*SupportedLanguages != 0) {
    for (Index = 0; SupportedLanguages[Index] != 0 && SupportedLanguages[Index] != ';'; Index++) {
    }

    if ((AsciiStrnCmp (SupportedLanguages, TargetLanguage, Index) == 0) && (TargetLanguage[Index] == 0)) {
      return EFI_SUCCESS;
    }

    SupportedLanguages += Index;
    for ( ; *SupportedLanguages != 0 && *SupportedLanguages == ';'; SupportedLanguages++) {
    }
  }

  return EFI_UNSUPPORTED;
}

/**
  Find the string for Language in UnicodeStringTable. Language must also be
  listed in SupportedLanguages, which uses ISO 639-2 codes when
  Iso639Language is TRUE and RFC 4646 codes otherwise.
**/
STATIC
EFI_STATUS
LookupUnicodeString2 (
  IN CONST CHAR8                     *Language,
  IN CONST CHAR8                     *SupportedLanguages,
  IN CONST EFI_UNICODE_STRING_TABLE  *UnicodeStringTable,
  OUT CHAR16                         **UnicodeString,
  IN BOOLEAN                         Iso639Language
  )
{
  BOOLEAN  Found;
  UINTN    Index;
  CHAR8    *LanguageString;

  if ((Language == NULL) || (UnicodeString == NULL)) {
    return EFI_INVALID_PARAMETER;
  }

  if ((SupportedLanguages == NULL) || (UnicodeStringTable == NULL)) {
    return EFI_UNSUPPORTED;
  }

  Found = FALSE;
  if (Iso639Language) {
    while (*SupportedLanguages != 0) {
      if (CompareIso639LanguageCode (Language, SupportedLanguages)) {
        Found = TRUE;
        break;
      }

      SupportedLanguages += 3;
    }
  } else {
    //
    // The arguments are passed in this order, so the list that is split at
    // semicolons is Language, and SupportedLanguages must equal one of its
    // elements.
    //
    Found = !IsLanguageSupported (Language, SupportedLanguages);
  }

  if (!Found) {
    return EFI_UNSUPPORTED;
  }

  //
  // Each table entry lists its languages separated by semicolons.
  //
  while (UnicodeStringTable->Language != NULL) {
    LanguageString = UnicodeStringTable->Language;
    while (0 != *LanguageString) {
      for (Index = 0; LanguageString[Index] != 0 && LanguageString[Index] != ';'; Index++) {
      }

      if (AsciiStrnCmp (LanguageString, Language, Index) == 0) {
        *UnicodeString = UnicodeStringTable->UnicodeString;
        return EFI_SUCCESS;
      }

      LanguageString += Index;
      for (Index = 0; LanguageString[Index] != 0 && LanguageString[Index] == ';'; Index++) {
      }
    }

    UnicodeStringTable++;
  }

  return EFI_UNSUPPORTED;
}

/**
  Test whether ControllerHandle is already managed by the driver that owns
  DriverBindingHandle through ProtocolGuid.
**/
STATIC
EFI_STATUS
EfiTestManagedDevice (
  IN CONST EFI_HANDLE  ControllerHandle,
  IN CONST EFI_HANDLE  DriverBindingHandle,
  IN CONST EFI_GUID    *ProtocolGuid
  )
{
  EFI_STATUS  Status;
  VOID        *ManagedInterface;

  Status = gBS->OpenProtocol (
                  ControllerHandle,
                  (EFI_GUID *)ProtocolGuid,
                  &ManagedInterface,
                  DriverBindingHandle,
                  ControllerHandle,
                  EFI_OPEN_PROTOCOL_BY_DRIVER
                  );
  if (!EFI_ERROR (Status)) {
    gBS->CloseProtocol (
           ControllerHandle,
           (EFI_GUID *)ProtocolGuid,
           DriverBindingHandle,
           ControllerHandle
           );
    return EFI_UNSUPPORTED;
  }

  if (Status != EFI_ALREADY_STARTED) {
    return EFI_UNSUPPORTED;
  }

  return EFI_SUCCESS;
}

EFI_STATUS
EFIAPI
ReimsVgpuComponentNameGetDriverName (
  IN  EFI_COMPONENT_NAME_PROTOCOL  *This,
  IN  CHAR8                        *Language,
  OUT CHAR16                       **DriverName
  )
{
  return LookupUnicodeString2 (
           Language,
           This->SupportedLanguages,
           mReimsVgpuDriverNameTable,
           DriverName,
           (BOOLEAN)(This == &gReimsVgpuComponentName)
           );
}

EFI_STATUS
EFIAPI
ReimsVgpuComponentNameGetControllerName (
  IN  EFI_COMPONENT_NAME_PROTOCOL  *This,
  IN  EFI_HANDLE                   ControllerHandle,
  IN  EFI_HANDLE                   ChildHandle        OPTIONAL,
  IN  CHAR8                        *Language,
  OUT CHAR16                       **ControllerName
  )
{
  EFI_STATUS  Status;

  //
  // Only the controller itself has a name.
  //
  if (ChildHandle != NULL) {
    return EFI_UNSUPPORTED;
  }

  Status = EfiTestManagedDevice (
             ControllerHandle,
             gReimsVgpuDriverBinding.DriverBindingHandle,
             &gEfiPciIoProtocolGuid
             );
  if (EFI_ERROR (Status)) {
    return Status;
  }

  return LookupUnicodeString2 (
           Language,
           This->SupportedLanguages,
           mReimsVgpuControllerNameTable,
           ControllerName,
           (BOOLEAN)(This == &gReimsVgpuComponentName)
           );
}
