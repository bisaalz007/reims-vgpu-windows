# The option ROM: the PCI device's UEFI driver (src/), linked as an x86-64
# EFI boot-service driver and packaged as a PCI option ROM
# (docs/architecture.md 11.4).
#
#   make        build/ReimsVgpu.efi and build/ReimsVgpu.rom
#   make clean  remove build/
#
# BUILD moves the output: cargo xtask dist runs `make rom` with BUILD set to
# <target>/firmware. Needs clang and lld (on macOS, Homebrew's: brew install
# lld). The ROM's bytes are the compiler's: Xcode's clang and Linux's build
# different ones from the same sources.

CC    = clang
LD    = lld-link
ROMLD = ld.lld
OBJCOPY = llvm-objcopy

BUILD = build
NAME  = ReimsVgpu

SOURCES = \
	src/ReimsVgpu.c \
	src/GraphicsOutput.c \
	src/ComponentName.c

TARGET = --target=x86_64-unknown-windows

CFLAGS = $(TARGET) -Os -ffreestanding -fshort-wchar -funsigned-char \
	-fno-ms-extensions -fno-stack-protector -mno-stack-arg-probe \
	-mno-red-zone -mno-implicit-float -fno-asynchronous-unwind-tables \
	-ffunction-sections -fdata-sections \
	-ftrap-function=undefined_behavior_has_been_optimized_away_by_clang \
	-Wall -Wextra -Werror -Wno-unused-parameter

LDFLAGS = /nologo /nodefaultlib /dll /driver /base:0 /machine:x64 \
	/subsystem:efi_boot_service_driver /entry:ReimsVgpuEntryPoint \
	/opt:ref /align:32 /filealign:32 /Brepro

OBJECTS = $(patsubst src/%,$(BUILD)/obj/%.obj,$(basename $(SOURCES)))

all: rom

efi: $(BUILD)/$(NAME).efi

rom: $(BUILD)/$(NAME).rom

# OptionRom.S lays out the ROM header around the linked image; ld.lld
# flattens it to raw bytes.
$(BUILD)/$(NAME).rom: src/OptionRom.S src/ReimsVgpu.h $(BUILD)/$(NAME).efi
	$(CC) --target=x86_64-unknown-none-elf -DEFI_IMAGE='"$(BUILD)/$(NAME).efi"' \
		-c $< -o $(BUILD)/obj/OptionRom.o
ifeq ($(OS),Windows_NT)
	$(OBJCOPY) -O binary $(BUILD)/obj/OptionRom.o $@
else
	$(ROMLD) --oformat binary -o $@ $(BUILD)/obj/OptionRom.o
endif

$(BUILD)/$(NAME).efi: $(OBJECTS)
	$(LD) $(LDFLAGS) /out:$@ $^

$(BUILD)/obj/%.obj: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all efi rom clean

-include $(OBJECTS:.obj=.d)
