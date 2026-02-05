"""
EOG Product - メインエントリーポイント
コマンドラインからキー入力を受け付け、対応する処理を実行する
"""


def acquisition():
    """データ取得処理"""
    print("acquisition を実行します...")
    # TODO: 実際の処理を実装
    # from scripts.acquisition import main
    # main()


def tabulation():
    """集計処理"""
    print("tabulation を実行します...")
    # TODO: 実際の処理を実装
    # from scripts.tabulation import main
    # main()


def show_help():
    """ヘルプを表示"""
    print("\n=== 利用可能なコマンド ===")
    print("  1 : データ取得 (acquisition)")
    print("  2 : 集計処理 (tabulation)")
    print("  h : ヘルプを表示")
    print("  q : 終了")
    print("=" * 28)


# キー入力と関数のマッピング
KEY_ACTIONS = {
    "1": acquisition,
    "2": tabulation,
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
