# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import csv
from pathlib import Path

import datasets
from PIL import Image

_CITATION = """
@inproceedings{fersini2022semeval,
title={SemEval-2022 Task 5: Multimedia automatic misogyny identification},
author={Fersini, Elisabetta and Gasparini, Francesca and Rizzi, Giulia and Saibene, Aurora and Chulvi, Berta and Rosso, Paolo and Lees, Alyssa and Sorensen, Jeffrey},
booktitle={Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)},
pages={533--549},
year={2022}
}
"""

_DESCRIPTION = """These are the datasets for Multimodal Misogyny Detection (MAMI), Task 5 of SemEval-2022."""


class SemEval2022Task5(datasets.GeneratorBasedBuilder):
    """These are the datasets for Multimodal Misogyny Detection (MAMI), Task 5 of SemEval-2022."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "figid": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "image": datasets.Image(),
                    "task-A": datasets.Value("int32"),
                    "task-B": datasets.Sequence(datasets.Value("string")),
                },
            ),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "images": Path("./mami2022", "training_images").resolve().as_posix(),
                    "metadata": Path("./mami2022", "training.csv").resolve().as_posix(),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "images": Path("./mami2022", "test_images").resolve().as_posix(),
                    "metadata": Path("./mami2022", "test.csv").resolve().as_posix(),
                },
            ),
        ]

    def _generate_examples(self, images, metadata):
        with open(metadata, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)  # skip header
            for id_, row in enumerate(reader):
                image_path = Path(images, row[0])
                yield id_, {
                    "figid": Path(image_path).stem,
                    "text": row[6],
                    "image": Image.open(image_path).convert("RGB"),
                    "task-A": row[1],
                    "task-B": [header[i + 2] for i, r in enumerate(row[2:6]) if r == "1"],
                }


builder_instance = SemEval2022Task5()
builder_instance.download_and_prepare()
ds = builder_instance.as_dataset()
ds.push_to_hub("shijli/semeval2022-task5", private=True)
