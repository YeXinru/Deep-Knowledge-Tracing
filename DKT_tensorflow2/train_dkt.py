# encoding:utf-8
import argparse
import time
import sys
from TensorFlowDKT import *
from data_process import *
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score


def run(args):
    # process data
    seqs_by_student, num_skills = read_file(args.dataset)
    # print "--------seqs_by_student------num_skills"
    #print[len(seqs_by_student), num_skills]
    train_seqs, test_seqs = split_dataset(seqs_by_student)
    train_seqs = generate_split_train(train_seqs,20)
    print("seqs_num:",len(train_seqs))
    batch_size = 10
    train_generator = DataGenerator(
        train_seqs, batch_size=batch_size, num_skills=num_skills)
    test_generator = DataGenerator(
        test_seqs, batch_size=batch_size, num_skills=num_skills)
    #print[len(test_seqs), num_skills]

    # config and create model
    config = {"hidden_neurons": [200],
              "batch_size": batch_size,
              "keep_prob": 0.6,
              "num_skills": num_skills,
              "input_size": num_skills * 2}
    model = TensorFlowDKT(config)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    lr = 0.4
    lr_decay = 0.92
    # run epoch
    for epoch in range(1):
        # train
        model.assign_lr(sess, lr * lr_decay ** epoch)
        overall_loss = 0
        train_generator.shuffle()
        st = time.time()
        while not train_generator.end:
            input_x, target_id, target_correctness, seqs_len, max_len = train_generator.next_batch()
            
            each_loss = model.step(sess, input_x, target_id,
                                       target_correctness, seqs_len, is_train=True)
            overall_loss += each_loss
            print("\r idx:{0}, each_loss:{1}, time spent:{2}s".format(train_generator.pos, each_loss,time.time() - st),sys.stdout.flush())

        # test
        test_generator.reset()
        preds, binary_preds, targets = list(), list(), list()
        while not test_generator.end:
            input_x, target_id, target_correctness, seqs_len, max_len = test_generator.next_batch()
            binary_pred, pred, _ = model.step(sess, input_x, target_id, target_correctness, seqs_len, is_train=False)
            for seq_idx, seq_len in enumerate(seqs_len):
                preds.append(pred[seq_idx, 0:seq_len])
                binary_preds.append(binary_pred[seq_idx, 0:seq_len])
                targets.append(target_correctness[seq_idx, 0:seq_len])
        # compute metrics
        preds = np.concatenate(preds)
        binary_preds = np.concatenate(binary_preds)
        targets = np.concatenate(targets)
        auc_value = roc_auc_score(targets, preds)
        accuracy = accuracy_score(targets, binary_preds)
        precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)
        print('\n auc={0}, accuracy={1}, precision={2}, recall={3}'.format(auc_value, accuracy, precision, recall))
        # 输出的猜测值
    #print "-------------猜测值-----------------\n"
    #print[preds, binary_pred, targets]
    #print model.current_state
    #print[model.output_w, model.output_b]


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="train dkt model")
    arg_parser.add_argument("--dataset", dest="dataset", required=True)
    args = arg_parser.parse_args()
    run(args)
