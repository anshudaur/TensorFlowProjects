import argparse
import pickle
from math import ceil

import tensorflow as tf


def files_to_tfrecord_fixedlen(*files, out_path, seq_len=200, overlap=0):
    """Process a number of text files into TFRecords data file.

    All files are conjoined into one big string. For simplicity, we split this
    string into equal-length sequences of seq_len-1 characters each.
    Furthermore, a special "beginning-of-sequence" character is prepended to
    each sequence, and the characters are mapped to integer indices
    representing one-hot vectors. We store the processed sequences into a
    TFrecords file; we also store the character-index mapping (vocabulary).

    Parameters:
        files: Paths to the text files to use for the corpus.
        out_path: Path to store the processed corpus, *without* file extension!
        seq_len: Requested sequence length.
        overlap: Float between 0 and 1. How much overlap there should be
                 between sequences. E.g. with a seq_len of 200 and an overlap
                 of 0.1, we only advance 180 characters between successive
                 sequences. Rounded down.
    """
    if not 0 <= overlap < 1:
        raise ValueError("Invalid overlap specified: {}. Please use a number "
                         "between 0 (inclusive) and 1 (exclusive).")

    full_text = "\n".join(open(file).read() for file in files)
    # we create a mapping from characters to integers, including a special
    # "beginning of sequence" character
    chars = set(full_text)
    ch_to_ind = dict(zip(chars, range(1, len(chars)+1)))
    ch_to_ind["<S>"] = 0

    seqs = text_to_seqs(full_text, seq_len, ch_to_ind, overlap)
    print("Split input into {} sequences...".format(len(seqs)))

    with tf.io.TFRecordWriter(out_path + ".tfrecords") as writer:
        for ind, seq in enumerate(seqs):
            tfex = tf.train.Example(features=tf.train.Features(feature={
                "seq": tf.train.Feature(int64_list=tf.train.Int64List(value=seq))
            }))
            writer.write(tfex.SerializeToString())
            if (ind + 1) % 100 == 0:
                print("Serialized {} sequences...".format(ind+1))
    pickle.dump(ch_to_ind, open(out_path + "_vocab", mode="wb"))


def text_to_seqs(text, seq_len, mapping, overlap):
    """Convert a string to a list of lists of equal length.

    Each character is mapped to its index as given by the mapping parameter.
    Right now this will actually use sequences *one character shorter* than
    requested, but prepend a "beginning of sequence" character.

    Parameters:
        text: String, the corpus.
        seq_len: Requested sequence length. See note above.
        mapping: Dict mapping characters to indices.
        overlap: Float between 0 and 1. How much overlap there should be
                 between sequences. E.g. with a seq_len of 200 and an overlap
                 of 0.1, we only advance 180 characters between successive
                 sequences. Rounded up.

    Returns:
        List of split character-index sequences.
    """
    use_bos = True
    if use_bos:
        seq_len -= 1

    steps_to_advance = seq_len - int(ceil(overlap * seq_len))

    seqs = [[mapping["<S>"]] + chs_to_inds(text[ind:(ind+seq_len)], mapping)
            for ind in range(0, len(text), steps_to_advance)]
    # we throw away any sequences that ended up shorter (usually at the very end)
    return [seq for seq in seqs if len(seq) == len(seqs[0])]


def chs_to_inds(char_list, mapping):
    """Helper to convert a list of characters to a list of corresponding indices.

    Parameters:
        char_list: List of characters (or string).
        mapping: Dict mapping characters to indices.

    Returns:
        List of character indices.
    """
    return [mapping[ch] for ch in char_list]


def parse_seq(example_proto, seq_len):
    """
    Needed to read the stored .tfrecords data -- import this in your
    training script.

    Parameters:
        example_proto: Protocol buffer of single example.
        seq_len: The sequence length corresponding to the example.

    Returns:
        Tensor containing the parsed sequence.
    """
    features = {"seq": tf.io.FixedLenFeature((seq_len,), tf.int64)}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return tf.cast(parsed_features["seq"], tf.int32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_files",
                        help="File paths to use as input, separated by commas."
                             " E.g. 'file1.txt,file2.txt'.")
    parser.add_argument("out_path",
                        help="Path to store the data to. Do *not* specify the "
                             "file extension, as this script stores both a "
                             ".tfrecords file as well as a vocabulary file.")
    parser.add_argument("-l", "--seqlen",
                        type=int,
                        default=200,
                        help="How many characters per sequence. Default: 200.")
    parser.add_argument("-o", "--overlap",
                        type=float,
                        default=0.,
                        help="Overlap between successive sequences, as a "
                             "fraction. Between 0 (inclusive) and 1 "
                             "(exclusive). Default: 0 (no overlap).")
    args = parser.parse_args()
    file_list = args.data_files.split(",")
    files_to_tfrecord_fixedlen(*file_list, out_path=args.out_path,
                               seq_len=args.seqlen, overlap=args.overlap)
