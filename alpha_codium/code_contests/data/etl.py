import os.path

from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.config_loader import get_settings


def etl(
    source=CodeContestDataProvider.hf_dataset_name,
    output_dataset_name=None,
    train_sample=0.1,
    translate_references=True,
    filter_languages=["PYTHON3"],  # noqa: B006
    disable_filter_languages=False,
):
    cc = CodeContestDataProvider(source)
    ds = cc.dataset
    if train_sample == 0:
        del ds["train"]
    elif train_sample < 1:
        ds["train"] = CodeContestDataProvider.sample(ds["train"], fraction=train_sample)

    if translate_references:
        ds = cc.translate_references(ds)

    if (not disable_filter_languages) and filter_languages and len(filter_languages):
        ds = cc.filter_solution_by_languages(ds=ds, languages=filter_languages)

    if output_dataset_name:
        base_path = os.path.expanduser(get_settings().etl.private_dataset_cache_dir)
        output_path = os.path.join(base_path, output_dataset_name)
        print(f"saving the output dataset to {output_path}")
        ds.save_to_disk(output_path)
    return ds


if __name__ == "__main__":
    etl(output_dataset_name="train_sample_python_only")


# data
# etl
# --source
# /Users/talrid/Desktop/code_contest_dataset/original
# --output_dataset_name
# valid_and_test
# --train_sample=0
# --filter_languages=''