import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont
from MAdvance import MAdvance

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h), color=(255,255,255))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

dataset_name='ffhq' #only support ffhq
M=MAdvance(dataset_name=dataset_name)

with open(M.img_path+'grad_example', 'rb') as handle:
    grads = pickle.load(handle)

img_index='0' #@param [0,1,2,3,4,5,6,7,8,9] 
img_index=int(img_index)
LayerIndex_ChannelIndex='12_414' #@param ['12_266','11_286','3_169','6_83','6_501','15_45','9_409','9_376','8_28'] 

list_of_images = []
list_strength = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
for strength in list_strength:
    print(strength)
    manipulation_strength=f'{strength}' #@param [3, 5, 10, 15] 
    alpha=int(manipulation_strength)

    tmp=LayerIndex_ChannelIndex.split('_')
    lindex,cindex=int(tmp[0]),int(tmp[1])
    grad=grads[lindex][img_index,cindex]
    tmp=np.zeros([32,32,3]).astype('uint8')
    tmp[:,:,1]=grad
    grad1=Image.fromarray(tmp).resize((512,512))

    M.alpha=[0]
    M.img_index=img_index
    M.num_images=1
    M.manipulate_layers=[lindex]
    codes,out=M.EditOneC(cindex) 
    original=Image.fromarray(out[0,0]).resize((512,512))

    original = original.convert("RGBA")
    grad1 = grad1.convert("RGBA")
    new_img = Image.blend(original, grad1, 0.5)

    M.alpha=[-alpha,alpha]
    M.manipulate_layers=[lindex]
    codes,out=M.EditOneC(cindex) 
    positive=Image.fromarray(out[0,1]).resize((512,512))
    negative=Image.fromarray(out[0,0]).resize((512,512))


    draw = ImageDraw.Draw(original)
    font = ImageFont.truetype("newsserifbold.ttf", 32)
    draw.text((0, 15), 'original', (255, 0, 0), font = font)

    draw = ImageDraw.Draw(new_img)
    font = ImageFont.truetype("newsserifbold.ttf", 32)
    draw.text((0, 15), 'gradient', (255, 0, 0), font = font)

    draw = ImageDraw.Draw(positive)
    font = ImageFont.truetype("newsserifbold.ttf", 32)
    draw.text((0, 15), 'positive manipulation {}'.format(strength), (255, 0, 0), font = font)

    draw = ImageDraw.Draw(negative)
    font = ImageFont.truetype("newsserifbold.ttf", 32)
    draw.text((0, 15), 'manipulation {}'.format(strength), (255, 0, 0), font = font)

    list_of_images.append(original)
    list_of_images.append(new_img)
    list_of_images.append(positive)
    list_of_images.append(negative)

    # plt.figure(figsize=(20,5), dpi= 100)
    # plt.subplot(1,4,1)
    # plt.imshow(original)
    # plt.title('original')
    # plt.axis('off')
    # plt.subplot(1,4,2)
    # plt.imshow(new_img)
    # plt.title('gradient')
    # plt.axis('off')
    # plt.subplot(1,4,3)
    # plt.imshow(positive)
    # plt.title('positive manipulation')
    # plt.axis('off')
    # plt.subplot(1,4,4)
    # plt.imshow(negative)
    # plt.title('negative manipulation')
    # plt.axis('off')

    # plt.savefig('manipulation_strength'+str(manipulation_strength)+'_'+str(img_index)+'.png')

grid_image = image_grid(list_of_images, len(list_strength), 4)
grid_image.save("manipulation_strength_{}.jpg".format(LayerIndex_ChannelIndex), optimize=True, quality=75)

# all_images = []
# for strength in [-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
#     print(strength)
#     manipulation_strength=f'{strength}' #@param [3, 5, 10, 15] 
#     alpha=int(manipulation_strength)

#     tmp=LayerIndex_ChannelIndex.split('_')
#     lindex,cindex=int(tmp[0]),int(tmp[1])
#     grad=grads[lindex][img_index,cindex]
#     tmp=np.zeros([32,32,3]).astype('uint8')
#     tmp[:,:,1]=grad

#     M.alpha=[alpha]
#     M.img_index=img_index
#     M.num_images=1
#     M.manipulate_layers=[lindex]
#     codes,out=M.EditOneC(cindex) 
#     img = Image.fromarray(out[0,0])
#     img = img.convert("RGBA")
#     img.save('manipulation_strength'+str(manipulation_strength)+'_'+str(img_index)+'.png')
#     all_images.append(img)

# img, *imgs = all_images
# img.save(fp="manipulation_{}.gif".format(LayerIndex_ChannelIndex), format='GIF', append_images=imgs,
#          save_all=True, duration=200, loop=0)
