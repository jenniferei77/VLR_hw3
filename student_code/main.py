import argparse
from student_code.simple_baseline_experiment_runner import SimpleBaselineExperimentRunner
from student_code.coattention_experiment_runner import CoattentionNetExperimentRunner
import pdb

if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--model', type=str, choices=['simple', 'coattention'], default='simple')
    parser.add_argument('--train_image_dir', type=str)
    parser.add_argument('--train_question_path', type=str)
    parser.add_argument('--train_annotation_path', type=str)
    parser.add_argument('--test_image_dir', type=str)
    parser.add_argument('--test_question_path', type=str)
    parser.add_argument('--test_annotation_path', type=str)

    parser.add_argument('--loaded_question_corpus', type=str)
    parser.add_argument('--loaded_answer_corpus', type=str)
    parser.add_argument('--train_best_answers_filepath', type=str)
    parser.add_argument('--val_best_answers_filepath', type=str)

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_data_loader_workers', type=int, default=10)
    args = parser.parse_args()    

    if args.model == "simple":
        experiment_runner_class = SimpleBaselineExperimentRunner
    elif args.model == "coattention":
        experiment_runner_class = CoattentionNetExperimentRunner
    else:
        raise ModuleNotFoundError()

    experiment_runner = experiment_runner_class(train_image_dir=args.train_image_dir,
                                                train_question_path=args.train_question_path,
                                                train_annotation_path=args.train_annotation_path,
                                                test_image_dir=args.test_image_dir,
                                                test_question_path=args.test_question_path,
                                                test_annotation_path=args.test_annotation_path,
                                                loaded_question_corpus=args.loaded_question_corpus,
                                                loaded_answer_corpus=args.loaded_answer_corpus,
                                                train_best_answers_filepath=args.train_best_answers_filepath,
                                                val_best_answers_filepath=args.val_best_answers_filepath,
                                                batch_size=args.batch_size,
                                                num_epochs=args.num_epochs,
                                                num_data_loader_workers=args.num_data_loader_workers)
    experiment_runner.train()
