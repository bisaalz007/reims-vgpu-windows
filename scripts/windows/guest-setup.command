#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
account=$(id -un)
[[ "$account" =~ ^[a-z_][a-z0-9_-]*$ ]] || { echo 'Unsupported account name'; exit 1; }
echo 'Reims: SSH による VM 操作、日本語キーボード、描画設定を準備します。'
sudo -v
mkdir -p "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
key=$(cat authorized_key)
grep -Fqx -- "$key" "$HOME/.ssh/authorized_keys" || printf '%s\n' "$key" >> "$HOME/.ssh/authorized_keys"
chmod 700 "$HOME/.ssh"
chmod 600 "$HOME/.ssh/authorized_keys"
sudo defaults write /Library/Preferences/.GlobalPreferences METAL_ERROR_MODE -int 5
sudo chmod 644 /Library/Preferences/.GlobalPreferences.plist
sudo defaults write /Library/Preferences/com.apple.keyboardtype keyboardtype -dict '1-1575-0' -int 42
sudo chmod 644 /Library/Preferences/com.apple.keyboardtype.plist
sudo /usr/sbin/systemsetup -settimezone Asia/Tokyo
# WHPX/PVG cannot reliably restore the guest's graphics state after ACPI sleep.
sudo /usr/bin/pmset -a sleep 0 disksleep 0 standby 0 powernap 0 hibernatemode 0
sudo /usr/bin/ssh-keygen -A
sudo launchctl enable system/com.openssh.sshd
sudo launchctl bootstrap system /System/Library/LaunchDaemons/ssh.plist 2>/dev/null || sudo launchctl kickstart system/com.openssh.sshd
rule=$(mktemp)
trap 'rm -f "$rule"' EXIT
printf '%s ALL=(root) NOPASSWD: /sbin/shutdown -h now, /sbin/shutdown -r now\n' "$account" > "$rule"
sudo /usr/sbin/visudo -cf "$rule"
sudo install -o root -g wheel -m 0440 "$rule" /etc/sudoers.d/reims-vm-control
echo
echo "Windows 管理ソフトの接続タブに入力してください。ユーザー名: $account"
sudo /usr/bin/ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub -E sha256
echo '設定の反映には macOS の再起動が必要です。Apple Account はシステム設定から本人がサインインしてください。'
read -r -p 'Enter で終了します。'
