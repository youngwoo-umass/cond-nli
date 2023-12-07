

def main():
    # run_list = ["exact_match", "gpt-3.5-turbo"]
    run_list = ["exact_match", "gpt-3.5-turbo", "davinci"]
    # run_list = ["slr"]
    tag_list = ["mismatch", "conflict"]
    for run_name in run_list:
        for tag in tag_list:
            print(run_name, tag)
            try:
                build_save(run_name, tag, 0.5)
            except FileNotFoundError as e:
                print(e)


if __name__ == "__main__":
    main()
