# Windows / WHPX の Apple Account サインイン

2026-10-13、macOS Tahoe 26.4（25E246 / Darwin 25.4.0）で、ユーザー本人が
Apple Account へのログイン成功を確認した。ネットワークは NAT のまま。

## 必要だった修正

1. OpenCore で VM 固有のシリアル・MLB・ROM・SystemUUID を設定する。
   この VM では既存のモデルと SystemUUID を保持した。SMBIOS のシリアル・MLB、
   EFI の system-id、Apple GUID 付き NVRAM の ROM/MLB/HW_ROM/HW_MLB が一致し、
   en0 が内蔵扱いであることを確認した。通常の `nvram -p` 一覧だけでは
   GUID 付きの値が見えないため、「存在しない」と判断しない。
2. `-accel whpx,hyperv=off` でタイマー周波数を正しく伝える。
   Hyper-V enlightenment の既定経路では 6.4 GHz と認識し、実測で時計が
   半分近い速さになっていた。変更後は 3187201000 Hz でホスト時刻と一致した。
3. OpenCore のカーネル互換設定で、アプリからの VM 判定経路を変更する。
   CPU の hypervisor ビットは維持し、`kern.hv_vmm_present` と
   `kern.hibernatecount` の sysctl 名を交換する。休止していない VM では
   前者が 0 を返す。CPU 自体の `hypervisor=off` はこのホストで初期起動が
   停止したため採用していない。

互換設定の名前交換は [公開されている OpenCore 設定例](https://gist.github.com/akshaynexus/5b9eaf8217949e5bc0f754e2d29aae39)
と同じ方式。ここでは x86_64 / Darwin **25.4.0 だけ**に限定し、各検索は
1 箇所だけを変更する。稼働中の BootKernelExtensions.kc に各パターンが
1 件ずつあることと、再起動後の sysctl 値を確認した。Apple のカーネルや
バイナリは配布しない。SIP・AMFI の設定は維持する。

**休止状態は使わない。** この方法では `kern.hibernatecount` が本来の
休止回数ではなくなる。macOS 更新後の別カーネルには自動適用しない。
対応範囲を広げる前にパターン、起動、認証を再確認する。

## 管理ソフトと CLI

VM を完全終了してから、管理ソフトの Apple Account 互換設定を適用／解除する。
CLI では次を使う。

```powershell
./scripts/windows/prepare-opencore.ps1 -AppleAccountCompatibility Enable -Rebuild
./scripts/windows/prepare-opencore.ps1 -AppleAccountCompatibility Disable -Rebuild
```

変更前の plist は `.work/diagnostics/opencore-before-account-*.plist` に保持する。
候補を `ocvalidate` で検証してから置き換える。既存の識別情報や他のカーネル
設定は保持し、再適用で同じエントリーを重複させない。通常の Rebuild は
互換設定をそのまま保持する。パスワードと二要素認証は macOS で本人が入力する。

## 失敗と成功を区別した記録

変更前は Anisette の登録開始が HTTP 200、登録完了が HTTP 401 だった。
`AKAnisetteError -8008`、ADI routing の `-45061` が続いていた。
時計修正だけでは同じ拒否が残った。

sysctl 互換設定後は登録完了も HTTP 200 となり、`ADIProvisioningEnd succeeded`
を確認した。その後、本人によるログイン成功の確認を受けた。
`DeviceIdentity` / BAA `-10000` / `AKAuthenticationError -7066` は成功時にも
ログに残るため、それだけをログイン不能の証拠にしてはいけない。

[Apple 公式の VM 向け iCloud 対応](https://developer.apple.com/documentation/virtualization/using-icloud-with-macos-virtual-machines)
は Apple Silicon ホストの Virtualization.framework と Secure Enclave を
利用する別の構成を説明している。この実装がそれを提供したという意味ではない。

NAT でも今回の認証は成功した。ブリッジへの変更、識別情報の再生成、
キーチェーンの削除は今回の成功に必要なかった。シリアルやトークン、
アカウント名を含む生ログは公開しない。
