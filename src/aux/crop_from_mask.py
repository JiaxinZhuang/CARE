import SimpleITK as sitk
from PIL import Image
import pylab
from libtiff import TIFF
import os


class Crop_from_mask:
    def read_jpg(self, path):
        img = Image.open(path)
        return img

    def read_dcm(self, path):
        ds = sitk.ReadImage(path)
        img_array = sitk.GetArrayFromImage(ds)
        img_bitmap = Image.fromarray(img_array[0])
        return img_bitmap

    def read_tif(self, path):
        tif = TIFF.open(path, mode='r')
        img = tif.read_image()
        return img

    def get_box(self, image):
        # (row,col) 为mask左上角坐标
        row = 0
        col = 0
        flag = False
        # print(image)
        for line in image:
            col = 0
            for x in line:
                if x != 0:
                    flag = True
                    break
                col += 1
            if flag == True:
                break
            row += 1

        print(row, col)

        line1 = image[row]
        line1 = line1[col:]

        count = 0

        for x in line1:
            if x != 0:
                count += 1
            else:
                break

        # (row1,col1)为mask右上角坐标
        row1 = row
        col1 = col + count - 1
        print(row1, col1)

        # (row2,col2)为mask右下角坐标
        row2 = row1
        col2 = col1
        count1 = 0
        while image[row2][col2] != 0:
            row2 += 1
            count1 += 1

        row2 -= 1
        print(row2, col2)
        return (col, row, col2, row2)


    """
    data_path 被裁剪图片的路径
    mask_path 标签图片的路径
    执行后保存裁剪后的图片
    """
    def crop_image(self, data_path, mask_path):
        *first, filename = data_path.split('/')
        shotname, extension = os.path.splitext(filename)
        if extension == 'dcm' or 'DCM':
            image = self.read_dcm(data_path)
        else:
            image = self.read_jpg(data_path)
        box = self.get_box(self.read_tif(mask_path))
        region = image.crop(box)
        region.save('./testphoto/'+shotname+'_crop'+'.png')

# croper = Crop_from_mask()
# croper.crop_image('./testphoto/111.dcm', './testphoto/Mask.tif')
# pylab.imshow(image)
# pylab.show()

# croper = Crop_from_mask()
# image =croper.read_jpg('./data/dog.png')
# region = image.crop((0,0,40,40))
# pylab.imshow(region)
# pylab.show()
