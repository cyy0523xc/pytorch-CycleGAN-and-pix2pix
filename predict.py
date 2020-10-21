"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from options.predict_options import PredictOptions
from data.image_dataset import ImageDataset
from models import create_model
from util.visualizer import save_images_only


if __name__ == '__main__':
    opt = PredictOptions().parse()  # get predict options
    # hard-code some parameters for predict
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    # 读取图像
    images = []
    src_dir = os.path.join(opt.dataroot, opt.phase)  # get the image directory
    assert os.path.isdir(src_dir)
    for fn in os.listdir(src_dir):
        if not fn.lower().endswith(('jpg', 'png', 'jpeg')):
            continue
        fn = os.path.join(src_dir, fn)
        images.append(Image.open(fn))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 数据集
    assert len(images) > 0
    dataloader = DataLoader(
        ImageDataset(images, transform),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.num_threads))

    # 保存目录
    image_dir = os.path.join(opt.results_dir, opt.name, opt.phase)
    if os.path.isdir(image_dir):
        print('result images directory: ', image_dir)
    else:
        print('creating result directory: ', image_dir)
        os.makedirs(image_dir)

    print('*'*40, epoch, '*'*40)
    start = time.time()
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        result = model.predict()        # run inference

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))

        # 保存图像
        print(result)
        print(result.shape)
        break
        # save_images_only(image_dir, visuals, img_path, aspect_ratio=opt.aspect_ratio)

    print('Time: ', time.time()-start)
