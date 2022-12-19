import sys
import io
import os
import yaml
import pickle
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

from scipy import misc

from model import get_embd
from eval.utils import calculate_roc, calculate_tar


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='build', help='model mode: build')
    parser.add_argument('--config_path', type=str, default='./configs/config_ms1m_100.yaml', help='config path, used when mode is build')
    parser.add_argument('--model_path', type=str, default='/data/hhd/InsightFace-tensorflow/output/20190116-130753/checkpoints/ckpt-m-116000', help='model path')
    parser.add_argument('--val_data', type=str, default='', help='val data, a dict with key as data name, value as data path')
    parser.add_argument('--train_mode', type=int, default=0, help='whether set train phase to True when getting embds. zero means False, one means True')
    parser.add_argument('--target_far', type=float, default=1e-3, help='target far when calculate tar')
        
    return parser.parse_args()


def load_bin(path, image_size):
    print('reading %s' % path)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    num = len(bins)
    images = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    images_f = np.zeros(shape=[num, image_size, image_size, 3], dtype=np.float32)
    # m = config['augment_margin']
    # s = int(m/2)

    cnt = 0
    for bin in bins:
        img = misc.imread(io.BytesIO(bin))
        img = misc.imresize(img, [image_size, image_size])
        # img = img[s:s+image_size, s:s+image_size, :]
        img_f = np.fliplr(img)
        img = img/127.5-1.0
        img_f = img_f/127.5-1.0
        images[cnt] = img
        images_f[cnt] = img_f
        cnt += 1
    print('done!')
    return (images, images_f, issame_list)

'''
# Bernardo
def load_pairs_samples_protocol_from_file(protocol_file_path='pairs.txt', dataset_path='', file_ext='.jpg'):
    pos_pair_label = True
    neg_pair_label = False
    all_pairs_paths_label = []

    with open(protocol_file_path, 'r') as fp:
        all_lines = [line.rstrip('\n') for line in fp.readlines()]
        # print('all_lines:', all_lines)
        num_folds, fold_size = int(all_lines[0].split('\t')[0]), int(all_lines[0].split('\t')[1])
        total_num_pairs = num_folds*(fold_size*2)
        
        global_pair_idx = 1
        while global_pair_idx < total_num_pairs:

            pos_pairs_paths = []
            for _ in range(1, fold_size+1):
                pos_pair = all_lines[global_pair_idx].split('\t')   # Aaron_Peirsol	1	2
                # print('pos_pair:', pos_pair)
                subj_name, index1, index2 = pos_pair
                assert index1 != index2
                path_sample1 = os.path.join(dataset_path, subj_name, subj_name+'_'+index1.zfill(4)+file_ext)
                path_sample2 = os.path.join(dataset_path, subj_name, subj_name+'_'+index2.zfill(4)+file_ext)
                pos_pair = (pos_pair_label, path_sample1, path_sample2)
                # print(str(global_pair_idx)+'/'+str(total_num_pairs) + ' - pos_pair:', pos_pair)
                # input('PAUSED')
                pos_pairs_paths.append(pos_pair)
                global_pair_idx += 1
                # print('path_sample1:', path_sample1)
                # print('path_sample2:', path_sample2)
                # print('pos_pair:', pos_pair)
            all_pairs_paths_label += pos_pairs_paths

            neg_pairs_paths = []
            for _ in range(1, fold_size+1):
                neg_pair = all_lines[global_pair_idx].split('\t')   # AJ_Cook	1	Marsha_Thomason	1
                # print('neg_pair:', neg_pair)
                subj_name1, index1, subj_name2, index2 = neg_pair
                assert subj_name1 != subj_name2
                path_sample1 = os.path.join(dataset_path, subj_name1, subj_name1+'_'+index1.zfill(4)+file_ext)
                path_sample2 = os.path.join(dataset_path, subj_name2, subj_name2+'_'+index2.zfill(4)+file_ext)
                neg_pair = (neg_pair_label, path_sample1, path_sample2)
                # print(str(global_pair_idx)+'/'+str(total_num_pairs) + ' - neg_pair:', neg_pair)
                # input('PAUSED')
                neg_pairs_paths.append(neg_pair)
                global_pair_idx += 1
            all_pairs_paths_label += neg_pairs_paths
                
        return all_pairs_paths_label
'''


def evaluate(embeddings, actual_issame, far_target=1e-3, distance_metric=0, nrof_folds=10):
    thresholds = np.arange(0, 4, 0.01)
    if distance_metric == 1:
        thresholds = np.arange(0, 1, 0.0025)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, dist = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), distance_metric=distance_metric, nrof_folds=nrof_folds)
    tar, tar_std, far = calculate_tar(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), far_target=far_target, distance_metric=distance_metric, nrof_folds=nrof_folds)
    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)
    # return tpr, fpr, acc_mean, acc_std, tar, tar_std, far       # original
    return tpr, fpr, acc_mean, acc_std, tar, tar_std, far, dist   # Bernardo


def run_embds(sess, images, batch_size, image_size, train_mode, embds_ph, image_ph, train_ph_dropout, train_ph_bn):
    if train_mode >= 1:
        train = True
    else:
        train = False
    batch_num = len(images)//batch_size
    left = len(images)%batch_size
    embds = []
    for i in range(batch_num):
        image_batch = images[i*batch_size: (i+1)*batch_size]
        cur_embd = sess.run(embds_ph, feed_dict={image_ph: image_batch, train_ph_dropout: train, train_ph_bn: train})
        embds += list(cur_embd)
        print('%d/%d' % (i, batch_num), end='\r')
    if left > 0:
        image_batch = np.zeros([batch_size, image_size, image_size, 3])
        image_batch[:left, :, :, :] = images[-left:]
        cur_embd = sess.run(embds_ph, feed_dict={image_ph: image_batch, train_ph_dropout: train, train_ph_bn: train})
        embds += list(cur_embd)[:left]
    print()
    print('done!')
    return np.array(embds)


if __name__ == '__main__':

    # Bernardo
    if not '--config_path' in sys.argv:
        # sys.argv += ['--config_path', './configs/config_ms1m_100_ms1mv2-1000subj.yaml']
        # sys.argv += ['--config_path', './configs/config_ms1m_100_ms1mv2-2000subj.yaml']
        sys.argv += ['--config_path', './configs/config_ms1m_100_ms1mv2-5000subj.yaml']

    if not '--model_path' in sys.argv:
        # sys.argv += ['--model_path', './output/arcface-resnet-v2-m-50_dataset=ms1mv2_1000classes_eval=lfw-calfw-etc_epoch=30_lr=0.01/checkpoints/ckpt-m-30000']
        # sys.argv += ['--model_path', './output/arcface-resnet-v2-m-50_dataset=ms1mv2_2000classes_eval=lfw-calfw-etc_epoch=30_lr=0.01/checkpoints/ckpt-m-30000']
        sys.argv += ['--model_path', './output/arcface-resnet-v2-m-50_dataset=ms1mv2_5000classes_eval=lfw-calfw-etc_epoch=30_lr=0.01/checkpoints/ckpt-m-30000']

    if not '--val_data' in sys.argv:
        # sys.argv += ['--val_data', '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/input/MS-Celeb-1M/faces_emore/lfw.bin']
        sys.argv += ['--val_data', '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/input/MS-Celeb-1M/faces_emore/calfw.bin']
        

    args = get_args()
    if args.mode == 'build':
        print('building...')
        config = yaml.load(open(args.config_path))
        images = tf.placeholder(dtype=tf.float32, shape=[None, config['image_size'], config['image_size'], 3], name='input_image')
        train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
        train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
        embds, _ = get_embd(images, train_phase_dropout, train_phase_bn, config)
        print('done!')
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            tf.global_variables_initializer().run()
            print('loading...')
            saver = tf.train.Saver()
            saver.restore(sess, args.model_path)
            print('done!')

            batch_size = config['batch_size']
            # batch_size = 32
            print('evaluating...')
            val_data = {}
            if args.val_data == '':
                val_data = config['val_data']
            else:
                val_data[os.path.basename(args.val_data)] = args.val_data
            for k, v in val_data.items():
                imgs, imgs_f, issame = load_bin(v, config['image_size'])

                print('forward running...')
                embds_arr = run_embds(sess, imgs, batch_size, config['image_size'], args.train_mode, embds, images, train_phase_dropout, train_phase_bn)
                embds_f_arr = run_embds(sess, imgs_f, batch_size, config['image_size'], args.train_mode, embds, images, train_phase_dropout, train_phase_bn)
                embds_arr = embds_arr/np.linalg.norm(embds_arr, axis=1, keepdims=True)+embds_f_arr/np.linalg.norm(embds_f_arr, axis=1, keepdims=True)
                print('done!')

                # tpr, fpr, acc_mean, acc_std, tar, tar_std, far = evaluate(embds_arr, issame, far_target=args.target_far, distance_metric=0, nrof_folds=10)      # original
                tpr, fpr, acc_mean, acc_std, tar, tar_std, far, dist = evaluate(embds_arr, issame, far_target=args.target_far, distance_metric=0, nrof_folds=10)  # Bernardo

                print('eval on %s: acc--%1.5f+-%1.5f, tar--%1.5f+-%1.5f@far=%1.5f' % (k, acc_mean, acc_std, tar, tar_std, far))

                path_pairs_distances = '/'.join(args.val_data.split('/')[:-1]) + '/' + args.val_data.split('/')[-1].split('.')[0] + '_distances_arcface='+str(config['class_num'])+'class_acc=%1.5f.npy' % (acc_mean)
                print('Bernardo')
                print('dist.shape:', dist.shape)
                print('Saving pairs distances:', path_pairs_distances)
                np.save(path_pairs_distances, dist)

            print('done!')
    else:
        raise ValueError("Invalid value for --mode.")

