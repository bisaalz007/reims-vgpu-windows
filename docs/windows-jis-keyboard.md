# Windows ホストの Apple JIS キーボード

2026-10-13 に Windows ネイティブ QEMU / WHPX と macOS Tahoe 26.4 で確認した、
英数・かな・ろ／アンダースコア・Command の修正記録。

## 症状と確認結果

通常の文字は入力できても、英数・かな・ろが届かなかった。
修飾キーを含めた問題に見えたが、Windows の入力処理と USB HID の記述に
別々の問題があった。現在は実際の Apple JIS キーボードで以下を確認済み。

| キー | 確認した動作 |
|---|---|
| 英数 | macOS の入力ソースが ABC に切り替わる |
| かな | 日本語ローマ字入力に切り替わる |
| Shift + ろ | アンダースコアを入力できる |
| Command | ゲストに修飾キーとして届く |

Windows のウィンドウから QEMU、仮想 USB キーボード、macOS という経路を確認した。
QMP から同じ言語キーを送った場合も、Carbon の現在の入力ソースが
`com.apple.inputmethod.Kotoeri.RomajiTyping.Japanese` と
`com.apple.keylayout.ABC` に切り替わることを別途確認した。

## 原因

### USB HID の Logical Maximum

QEMU のキーボードは論理最大値 255 を一バイトの `0xff` で記述していた。
macOS ゲストでは JIS の追加キーが受け付けられず、通常の文字だけが届いた。
最大値を正の二バイト値 `0x00ff` にすると、実物の「ろ」が届くようになった。

レポートの長さが 63 から 64 バイトになるため、フルスピード・ハイスピード双方の
HID ディスクリプターの長さも一緒に更新している。
修正箇所は `vendor/qemu/hw/usb/dev-hid.c`。

### QEMU の言語キー変換

QEMU の `hid_usage_keys` ではスキャンコード `0x71` / `0x72` の項目が空だった。
これを USB キーボードページの LANG2 (`0x91`) / LANG1 (`0x90`) に対応付けた。
Mac JIS ではそれぞれ英数・かなとして使われる。
修正箇所は `vendor/qemu/hw/input/hid.c`。

### Windows の IME キーと Raw Input

Windows 上の、フォーカスを持つ診断ウィンドウで次の信号を実測した。
文字列の入力は記録せず、特殊キーの番号だけを記録した。

| キー | Windows メッセージの仮想キー | メッセージのスキャンコード | Raw Input の make code |
|---|---|---|---|
| 英数 | `0x1a` | `0xf1` | `0x71` |
| かな | `0x16` | `0xf2` | `0x72` |

Raw Input は押下側でも break フラグを持ち、解放側には E1 フラグも付いていた。
SDL の Raw Input 処理では、通常のキー押下や解放として扱えない。
Raw Input と低レベルのキーボード捕捉を組み合わせると、修飾キーの配送も妨げる。

最終実装では Windows の通常メッセージ経路を使う。さらに、対象の SDL ウィンドウが
キーボードフォーカスを持つ場合に限り、実測した英数・かなの IME キーメッセージを
USB LANG2 / LANG1 の SDL イベントに変換する。押下・解放・リピートを区別し、
一般の文字や他のウィンドウの入力は変換しない。

実装は `crates/reims-vgpu-window/src/native/linux/windows.rs`。
Linux と共有する SDL バックエンドの配下にあるが、`cfg(windows)` で限定される。
`ui.rs` が SDL 初期化後にフックを登録する。

## macOS と Windows の設定

- ゲストの JIS 配列は `/Library/Preferences/com.apple.keyboardtype.plist` の
  `keyboardtype`、対象デバイス `1-1575-0` の値 `42` で確認した。
- `REIMSTOOLS/setup.command` はこの配列設定を含む。読み取り可能なファイル権限で保存する。
- 入力ソースには ABC と日本語ローマ字入力を登録しておく。
- 捕捉解除の表示は Windows 向けに `Ctrl+Alt+Win` とする。
  Apple キーボード上では Control・Option・Command に対応する。
- `REIMS_JIS_TRACE=1` は JIS と Command の位置だけを診断ログに出すための任意設定。
  通常の起動スクリプトはこの設定を有効にしない。

## 検証方法

1. ネイティブの `tests/unit/check-hid-jis.py` で、実際の HID デコーダーをビルドする。
   英数・かな・ろ・円記号・通常の文字・Command の押下／解放と、実際の HID
   ディスクリプターの範囲・長さを検証する。
2. `reims-vgpu-window` の 92 テストと clippy を実行する。
   特殊メッセージの変換、通常キーの非変換、フォーカスを失ったときの解放を含む。
3. 新しい QEMU とライブラリで macOS をコールドブートする。
4. QMP で言語キーを送り、ゲストの現在の入力ソースを確認する。
5. 実物のキーボードで四種類のキーを確認する。CLI 入力の成功だけで
   Windows の物理入力経路まで正常とは判断しない。

キーボード設定アシスタントが最前面にあると、テスト用メモではなく識別処理が
キーを受け取る。入力先を確認してから試すこと。
4K での描画負荷も入力不能と混同しやすいため、切り分けは既定の 1080p で行う。

診断ログ・スクリーンショット・ゲストの識別情報は、公開リポジトリには含めない。
