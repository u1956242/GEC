"""Main Program"""

import argparse
import os.path
import string
import datetime

from data import evaluation as ev
from data.reader import Dataset
from data.generator import DataGenerator

from lib.utils.report import report

from tool.transformer import Transformer

if __name__ == '__main__':
    # Arguments Parser
    parser = argparse.ArgumentParser()

    # Argument Generate: True for generating the data files from the HuggingFace's dataset
    parser.add_argument("--generate", action="store_true", default=False)

    # Argument Train: True for training a model
    parser.add_argument("--train", action="store_true", default=False)

    # Argument Test: True for testing a model
    parser.add_argument("--test", action="store_true", default=False)

    parser.add_argument("--norm_accentuation", action="store_true", default=False)
    parser.add_argument("--norm_punctuation", action="store_true", default=False)

    # Argument Mode: ["spelling"]
    parser.add_argument("--mode", type=str, default="spelling")

    args = parser.parse_args()

    max_text_length = 128
    charset_base = string.printable[:95]
    charset_special = """ÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÚÛÜÝàáâãäåçèéêëìíîïñòóôõöùúûüý"""
    batch_size = 1
    epochs = 5

    if args.generate:
        data = Dataset()
        data.read_dataset()

        if args.mode == "spelling":
            data.generate_spelling_data()
        elif args.mode == "spelling-words":
            data.generate_spelling_words_data()
        elif args.mode == "punctuation":
            data.generate_punctuation_data()

    else:

        train_file_name = ""
        test_file_name = ""
        output_path = ""
        target_path = ""

        if args.mode == "spelling":
            train_file_name = os.path.join("..", "data", args.mode, "spelling-train.txt")
            test_file_name = os.path.join("..", "data", args.mode, "spelling-test.txt")
            output_path = os.path.join("..", "output", args.mode)
            target_path = os.path.join(output_path, "checkpoint_weights.hdf5")

        elif args.mode == "spelling-words":
            train_file_name = os.path.join("..", "data", args.mode, "spelling-words-train.txt")
            test_file_name = os.path.join("..", "data", args.mode, "spelling-words-test.txt")
            output_path = os.path.join("..", "output", args.mode)
            target_path = os.path.join(output_path, "checkpoint_weights.hdf5")

        elif args.mode == "punctuation":
            train_file_name = os.path.join("..", "data", args.mode, "punctuation-train.txt")
            test_file_name = os.path.join("..", "data", args.mode, "punctuation-test.txt")
            output_path = os.path.join("..", "output", args.mode)
            target_path = os.path.join(output_path, "checkpoint_weights.hdf5")

        else:
            print("Mode not implemented")
            exit(-1)

        dtgen = DataGenerator(train_file_name=train_file_name,
                              test_file_name=test_file_name,
                              batch_size=batch_size,
                              charset=(charset_base + charset_special),
                              max_text_length=max_text_length,
                              predict=args.test,
                              max_sentances=1000,
                              items_per_sentance=None)

        dtgen.one_hot_process = False

        # Recommended
        model = Transformer(dtgen.tokenizer,
                            num_layers=6,
                            units=128,
                            d_model=64,
                            num_heads=8,
                            dropout=0.1,
                            stop_tolerance=20,
                            reduce_tolerance=15)

        # set `learning_rate` parameter or None for custom schedule learning
        model.compile(learning_rate=0.001)
        model.load_checkpoint(target=target_path)

        if args.train:
            model.summary(output_path, "summary.txt")
            callbacks = model.get_callbacks(logdir=output_path, checkpoint=target_path, verbose=1)

            start_time = datetime.datetime.now()

            h = model.fit(x=dtgen.next_train_batch(),
                          epochs=epochs,
                          steps_per_epoch=dtgen.steps['train'],
                          callbacks=callbacks,
                          shuffle=True,
                          verbose=1)

            total_time = datetime.datetime.now() - start_time

            loss = h.history['loss']
            accuracy = h.history['accuracy']

            time_epoch = (total_time / len(loss))
            total_item = (dtgen.size['train'])

            t_corpus = "\n".join([
                f"Total train sentences:      {dtgen.size['train']}",
                f"Batch:                      {dtgen.batch_size}\n",
                f"Total epochs:               {len(accuracy)}",
                f"Total time:                 {total_time}",
                f"Time per epoch:             {time_epoch}",
                f"Time per item:              {time_epoch / total_item}\n"
            ])

            with open(os.path.join(output_path, "train.txt"), "w") as lg:
                lg.write(t_corpus)
                print(t_corpus)

        elif args.test:
            start_time = datetime.datetime.now()

            predicts = model.predict(x=dtgen.next_test_batch(), steps=dtgen.steps['test'], verbose=1)

            total_time = datetime.datetime.now() - start_time

            old_metric, new_metric = ev.ocr_metrics(ground_truth=dtgen.dataset['test']['gt'],
                                                    data=dtgen.dataset['test']['dt'],
                                                    predict=predicts,
                                                    norm_accentuation=args.norm_accentuation,
                                                    norm_punctuation=args.norm_punctuation)

            p_corpus, e_corpus = report(dtgen=dtgen,
                                        predicts=predicts,
                                        metrics=[old_metric, new_metric],
                                        total_time=total_time)

            sufix = ("_norm" if args.norm_accentuation or args.norm_punctuation else "") + \
                    ("_accentuation" if args.norm_accentuation else "") + \
                    ("_punctuation" if args.norm_punctuation else "")

            with open(os.path.join(output_path, "predict.txt"), "w") as lg:
                lg.write("\n".join(p_corpus))
                print("\n".join(p_corpus[:30]))

            with open(os.path.join(output_path, f"evaluate{sufix}.txt"), "w") as lg:
                lg.write(e_corpus)
                print(e_corpus)