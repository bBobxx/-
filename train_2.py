from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import paddle
import paddle.fluid as fluid
import glob

def conv_bn_layer(input, ch_out, filter_size, stride, padding, activation='leaky_relu'):
    # combine conv2d and BN
    conv1 = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=activation)
    return conv1


def shortcut(input, ch_out, stride):
    # this is the same as shorcut in the resnet
    return conv_bn_layer(input, ch_out, 1, stride, 0, None)



def bottleneck(input, ch_out, stride, k_sz):
    # this is a bottleneck in a block,a block compromise several bottleneck, defined below
    short_cut = shortcut(input, ch_out*4, stride)
    conv1 = conv_bn_layer(input, ch_out, 1, stride, 0)
    conv20 = conv_bn_layer(conv1, ch_out, k_sz, 1, int((k_sz-1)/2))
    conv21 = conv_bn_layer(conv1, ch_out, k_sz-2, 1, int((k_sz-3)/2))
    conv22 = conv_bn_layer(conv1, ch_out, k_sz-4, 1, int((k_sz-5)/2))
    conv3 = fluid.layers.concat([conv1, conv20, conv21, conv22], axis=1)
    return fluid.layers.elementwise_add(x=short_cut, y=conv3, act='leaky_relu')


def block(func, input, ch_out, count, stride, k_sz):
    # this is the block, contains count bottleneck
    iter_bottleneck = func(input, ch_out, stride, k_sz)
    for i in range(1, count):
        iter_bottleneck = func(iter_bottleneck, ch_out, 1, k_sz)
    return iter_bottleneck

def multi_col(input, filter_size):
    conv1 = conv_bn_layer(input, ch_out=64, filter_size=filter_size, stride=2, padding=int((filter_size-1)/2))
    pool1 = fluid.layers.pool2d(input=conv1, pool_type='max', pool_size=3, pool_stride=2, pool_padding=1)
    res1 = block(bottleneck, pool1, 16, 1, 1, filter_size)
    res2 = block(bottleneck, res1, 16, 1, 2, filter_size)
    res3 = block(bottleneck, res2, 16, 1, 2, filter_size)
    res4 = block(bottleneck, res3,  16, 1, 2, filter_size)
    return res1, res2, res3, res4


def resnet_out(input):
    res01, res02, res03, res04 = multi_col(input, 9)
    res11, res12, res13, res14 = multi_col(input, 7)
    res21, res22, res23, res24 = multi_col(input, 5)
    resout1 = fluid.layers.concat([res01, res11, res21], axis=1)
    resout2 = fluid.layers.concat([res02, res12, res22], axis=1)
    resout3 = fluid.layers.concat([res03, res13, res23], axis=1)
    resout4 = fluid.layers.concat([res04, res14, res24], axis=1)
    return [resout1, resout2, resout3, resout4]


def group_net(resout):
    fl_sz = resout.shape
    fcn1 = conv_bn_layer(resout, 125, (fl_sz[-2], fl_sz[-1]), 1, 0)
    fcn2 = conv_bn_layer(fcn1, 10, 1, 1, 0)
    fcn2 = fluid.layers.reshape(fcn2, [-1, 10])
    softmax = fluid.layers.softmax(fcn2)
    return softmax


def FPN_and_groupout(input_im):
    in_sz = input_im.shape
    resnet_blocks = resnet_out(input_im)
    groupout = group_net(resnet_blocks[-1])
    pyramid_out = []
    last_fm = None
    for block in reversed(resnet_blocks):
        bridge = conv_bn_layer(block, 32, 1, 1, 0)
        if last_fm is not None:
            sz = bridge.shape
            upsample = fluid.layers.conv2d_transpose(last_fm, 32, stride=2,
                                                     output_size=(sz[-2], sz[-1]), act='relu')
            upsample = conv_bn_layer(upsample, 32, 1, 1, 0)
            last_fm = fluid.layers.elementwise_add(bridge, upsample)
        else:
            last_fm = bridge
        out = fluid.layers.conv2d(last_fm, 1, 1, 1, 0, act='relu')
        pyramid_out.append(out)
    pyramid_outone = fluid.layers.conv2d_transpose(pyramid_out[-1], 32, stride=2,
                                                   output_size=(int(in_sz[-2]/2), int(in_sz[-1]/2)), act='relu')
    pyramid_outone = fluid.layers.conv2d_transpose(pyramid_outone, 1, stride=2,
                                                   output_size=(in_sz[-2], in_sz[-1]),act='relu')
    return pyramid_outone, groupout


def create_group(num_ls):
    group_num = fluid.layers.one_hot(num_ls, 10)
    return group_num


def train():

    def save_model(postfix):
        model_path = os.path.join('./work', postfix)
        print ('save models to %s' % (model_path))
        fluid.io.save_params(exe, model_path)

    def network(is_train):
        record_file = glob.glob('./train*.recordio')
        test_file = glob.glob('./test*.recordio')
        file_obj = fluid.layers.open_files(
            filenames= record_file if is_train else test_file ,
            shapes = [[-1,3, 540, 960], [-1,1,540, 960],[-1, 1], [-1, 1]],
            dtypes=['float32','float32','int64', 'int64'],
            lod_levels=[0, 0, 0, 0],
            pass_num=100000
        )
        file_obj = fluid.layers.shuffle(file_obj, 500)
        file_obj = fluid.layers.batch(file_obj, batch_size=4 if is_train else 100)
        img, des_im, total_num, group_num = fluid.layers.read_file(file_obj)
        print('read over')# here is the data
        total_num = fluid.layers.cast(total_num, dtype="float32")
        group_num = create_group(group_num)
        predict1, predict0 = FPN_and_groupout(img)  # build our network
        delta0 = 100
        delta1 = 20
        loss0 = fluid.layers.elementwise_sub(predict1, des_im)
        loss0 = fluid.layers.reduce_mean(fluid.layers.abs(loss0))
        loss1 = fluid.layers.reduce_mean(fluid.layers.square_error_cost(
            input=fluid.layers.reduce_sum(predict1, dim=[2, 3]), label=total_num))
        loss2 = fluid.layers.cross_entropy(input=predict0, label=group_num, soft_label=True)
        loss2 = fluid.layers.reduce_mean(loss2)
        loss = fluid.layers.reduce_mean(fluid.layers.elementwise_add(
            fluid.layers.elementwise_add(loss0*delta0, loss1),delta1*loss2))
        # here if we only use loss0, then the final loss is about e-05,
        # and the error is about e+01, so we add a delta in the loss
        return loss, predict1, total_num
    with fluid.unique_name.guard():
        train_loss, pre_train, tr_num = network(is_train=True)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=
                                                  fluid.layers.exponential_decay(
                                                      0.0007, 4000, 0.9))
        optimizer.minimize(train_loss)
    test_program = fluid.Program()
    with fluid.unique_name.guard():
        with fluid.program_guard(test_program, fluid.Program()):
            loss, pre, true_num = network(is_train=False)
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    min_error = 1000
    for i in range(200000):
        loss_n, pretrain, tr_train = exe.run(
            program=fluid.default_main_program(), fetch_list=[train_loss.name, pre_train.name, tr_num.name])
        acc0 = np.abs(np.rint(np.sum(np.sum(pretrain, axis=-1), axis=-1))-np.rint(tr_train))/tr_train
        av_acc0 = np.sum(acc0)/np.shape(acc0)[0]
        print ("step {} train loss is {}, train error is {}".format(i, loss_n, av_acc0))
        if i%1000 == 0:
            #save_model(str(i))
            pre_map, tr_nums = exe.run(program=test_program, fetch_list=[pre.name, true_num.name])
            acc = np.abs(np.rint(np.sum(np.sum(pre_map, axis=-1), axis=-1))-np.rint(tr_nums))/tr_nums
            acc_mae = np.sum(np.abs(
                np.rint(np.sum(np.sum(pre_map,axis=-1),axis=-1))-np.rint(tr_nums)))/np.shape(acc)[0]
            av_acc = np.sum(acc)/np.shape(acc)[0]
            if av_acc < min_error:
                min_error = av_acc
                if i > 10000:
                    save_model('time2'+str(i))
            print("MAE is {}".format(acc_mae))
            print("min erro is {}".format(min_error))
            print("average error is {}".format(av_acc))
        if (i+1) % 10000 == 0:
            if i>10000 :
                save_model('time2' + str(i))

if __name__  == '__main__':
    train()
