"""
EOG Product - メインエントリーポイント
コマンドラインからキー入力を受け付け、対応する処理を実行する

実行方法:
    python main.py
"""


def get_labeled():
    """ラベル付きデータ取得"""
    print("ラベル付きデータ取得を開始します...")
    from scripts import get_labeled as gl
    gl.run()


def get_input():
    """入力データ取得"""
    print("入力データ取得を開始します...")
    from scripts import get_input as gi
    gi.run()


def generate_context():
    """教師データ生成"""
    print("教師データ生成を開始します...")
    from scripts import generate_context as gc
    gc.run()


def show_help():
    """ヘルプを表示"""
    print("\n=== 利用可能なコマンド ===")
    print("  l : ラベル付きデータ取得 (get_labeled)")
    print("  i : 入力データ取得 (get_input)")
    print("  c : 教師データ生成 (generate_context)")
    print("  h : ヘルプを表示")
    print("  q : 終了")
    print("=" * 28)


# キー入力と関数のマッピング
KEY_ACTIONS = {
    "l": get_labeled,
    "i": get_input,
    "c": generate_context,
    "h": show_help,
}


def main():
    """メインループ: キー入力を受け付けて対応する関数を実行"""
    print("EOG Product を起動しました")
    show_help()

    while True:
        try:
            user_input = input("\nコマンドを入力してください > ").strip().lower()

            if user_input == "q":
                print("終了します")
                break

            if user_input in KEY_ACTIONS:
                KEY_ACTIONS[user_input]()
            else:
                print(f"不明なコマンド: '{user_input}'")
                print("'h' でヘルプを表示します")

        except KeyboardInterrupt:
            print("\n終了します")
            break
        except EOFError:
            print("\n終了します")
            break


if __name__ == "__main__":
    main()
