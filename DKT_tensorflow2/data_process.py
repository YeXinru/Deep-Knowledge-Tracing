# encoding:utf-8
import numpy as np
import random

'''
def read_file(dataset_path):
    seqs_by_student = {}
    num_skills = 0
    with open(dataset_path, 'r') as f:
        for line in f:
            fields = line.strip().split()
            student, problem, is_correct = int(
                fields[0]), int(fields[1]), int(fields[2])
            num_skills = max(num_skills, problem)
            seqs_by_student[student] = seqs_by_student.get(
                student, []) + [[problem, is_correct]]
    return seqs_by_student, num_skills + 1
'''

def read_file(dataset_path):
    seqs_by_student = {}
    num_skills = 0
    dic={}
    a=0
    with open(dataset_path, 'r') as f:
        for line in f:
            fields = line.strip().split()
            student, problem, is_correct = int(
                fields[0]), int(fields[1]), int(fields[2])
            if problem in dic.keys():
                problem = dic[problem]
            else:
                dic[problem] = a
                problem = a
                a = a+1
            num_skills = max(num_skills, problem)
            seqs_by_student[student] = seqs_by_student.get(student, []) + [[problem, is_correct]]
    return seqs_by_student, num_skills + 1

#xinru:
def split_sequence(seqs,sequence_length):
    long = len(seqs) 
    res=[]
    gap=long-sequence_length
    if gap<=0:
        return [seqs]
    for i in range(0,gap):
        res_tmp = seqs[i:i+sequence_length]
        res.append(res_tmp)
    return res
def generate_split_train(train_seqs,sequence_length):
    train=[]
    for i in range(len(train_seqs)):
        train=train+(split_sequence(train_seqs[i],sequence_length))
    return train


def split_dataset(seqs_by_student, sample_rate=0.2, random_seed=1):
    sorted_keys = sorted(seqs_by_student.keys())
    random.seed(random_seed)
    test_keys = random.sample(sorted_keys, int(len(sorted_keys) * sample_rate))
    test_seqs = [seqs_by_student[k] for k in seqs_by_student if k in test_keys]
    train_seqs = [seqs_by_student[k]
                  for k in seqs_by_student if k not in test_keys]
    return train_seqs, test_seqs


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):

    lengths = [len(s) for s in sequences]
    # 10
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(
                'Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
'''
x(padding='post'):
[[  0   0   0 ...   0   0   0]
 [ 82  82  82 ...   0   0   0]
 [130 130   6 ... 247 247 247]
 ...
 [124 124 133 ...   0   0   0]
 [124 124 133 ...   0   0   0]
 [124 124 135 ...   0   0   0]]

x(padding='pre')
 [[  0   0   0 ...   0   0   0]
 [  0   0   0 ...  82  82  82]
 [130 130   6 ... 247 247 247]
 ...
 [  0   0   0 ... 138 138 138]
 [  0   0   0 ... 138 138 138]
 [  0   0   0 ... 124 124 135]]
'''


def num_to_one_hot(num, dim):
    base = np.zeros(dim)
    if num >= 0:
        base[num] += 1
    return base


def format_data(seqs, batch_size, num_skills):
    # seq<key,is_correct>
    gap = batch_size - len(seqs)

    seqs_in = seqs + [[[0, 0]]] * gap  # pad batch data to fix size

    #seq_len = np.array(map(lambda seq: len(seq), seqs_in)) - 1
    seq_len = np.array(list(map(lambda seq: len(seq)-1, seqs_in)))
 
    max_len = max(seq_len)
    #[:-1]removing last

    # max_len = 120(why subtract 1)

    x = pad_sequences(np.array([[(j[0] + num_skills * j[1]) for j in i[:-1]]
                                for i in seqs_in]), maxlen=max_len, padding='post', value=-1)
    input_x = np.array([[num_to_one_hot(j, num_skills * 2)
                         for j in i] for i in x])
    target_id = pad_sequences(np.array(
        [[j[0] for j in i[1:]] for i in seqs_in]), maxlen=max_len, padding='post', value=0)
    target_correctness = pad_sequences(np.array(
        [[j[1] for j in i[1:]] for i in seqs_in]), maxlen=max_len, padding='post', value=0)
    return input_x, target_id, target_correctness, seq_len, max_len

# batch_size=10


class DataGenerator(object):
    def __init__(self, seqs, batch_size, num_skills):
        self.seqs = seqs
        self.batch_size = batch_size
        self.pos = 0
        self.end = False 
        self.size = len(seqs)
        self.num_skills = num_skills

    def next_batch(self):
        batch_size = self.batch_size
        if self.pos + batch_size < self.size:
            batch_seqs = self.seqs[self.pos:self.pos + batch_size]
            self.pos += batch_size
        else:
            batch_seqs = self.seqs[self.pos:]
            self.pos = self.size - 1
        if self.pos >= self.size - 1:
            self.end = True
        input_x, target_id, target_correctness, seqs_len, max_len = format_data(
            batch_seqs, batch_size, self.num_skills)
        return input_x, target_id, target_correctness, seqs_len, max_len

    def shuffle(self):
        self.pos = 0
        self.end = False
        np.random.shuffle(self.seqs)

    def reset(self):
        self.pos = 0
        self.end = False


if __name__ == "__main__":
    seqs_by_student, num_skills = read_file('../data/assistments.txt')
    #seqs_by_student, num_skills = read_file('./data/testdata.txt')
    print (num_skills)
    train_seqs, test_seqs = split_dataset(seqs_by_student)
    print (len(train_seqs))
    batch_size = 10
    train_generator = DataGenerator(
        train_seqs, batch_size=batch_size, num_skills=num_skills)
    input_x, target_id, target_correctness, seqs_len, max_len = train_generator.next_batch()
    print([input_x.shape, target_id.shape,
          target_correctness, seqs_len, max_len])
    test_generator = DataGenerator(
        test_seqs, batch_size=batch_size, num_skills=num_skills)

#input_x (10,120,248) (batch_size,max_len_of_answerproblem,num_skills*2)
