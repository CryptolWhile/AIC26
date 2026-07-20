import numpy as np
import ffmpeg
from tqdm import tqdm

def get_frames(fn, width=48, height=27):
    video_stream, err = (
        ffmpeg
        .input(fn)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .run(capture_stdout=True, capture_stderr=True)
    )
    video = np.frombuffer(video_stream, np.uint8).reshape([-1, height, width, 3])
    return video

def visualize_predictions(frames, predictions=None, predictions_2=None, predictions_3=None, show_frame_num=False):
    from PIL import Image, ImageDraw, ImageFont

    if isinstance(predictions, np.ndarray):
        predictions = [predictions]
    if isinstance(predictions_2, np.ndarray):
        predictions_2 = [predictions_2]
    if isinstance(predictions_3, np.ndarray):
        predictions_3 = [predictions_3]

    ih, iw, ic = frames.shape[1:]
    width = 25

    # pad frames so that length of the video is divisible by width
    # pad frames also by len(predictions) pixels in width in order to show predictions
    pad_with = width - len(frames) % width if len(frames) % width != 0 else 0
    frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)])

    predictions = [np.pad(x, (0, pad_with)) for x in predictions]
    height = len(frames) // width

    img = frames.reshape([height, width, ih + 1, iw + len(predictions), ic])
    img_tmp = np.concatenate(np.split(
        np.concatenate(np.split(img, height), axis=2)[0], width
    ), axis=2)[0, :-1]
#     (1231, 1225, 3) 44 25 27 48
#     print(img_tmp.shape, height, width, ih, iw)

    img = Image.fromarray(img_tmp)
    draw = ImageDraw.Draw(img)
    
    if show_frame_num:
        font = ImageFont.truetype("/share/ai_platform/zhuwentao/times-ro.ttf", 12)
        # draw.text((x, y),"Sample Text",(r,g,b))
        for h in range(height):
            for w in range(width):
                avg_c = img_tmp[h * (ih + 1) + 3 : h * (ih + 1) + 9, w * (iw + 1) : w * (iw + 1)+12, :]
                avg_c = avg_c.sum()
                avg_c /= (3 * 6 * 12)
                n = h * width + w
                draw.text(
                    (
                        w * (iw + 1),
                        h * (ih + 1)+3
                    ),
                    str(n),
                    fill=(
                        255, # - img_tmp[h * (ih + 1) + 3, w * (iw + 1), 0],
                        255, # - img_tmp[h * (ih + 1) + 3, w * (iw + 1), 1],
                        255) if avg_c < 128 else (0, 0, 0), # - img_tmp[h * (ih + 1) + 3, w * (iw + 1), 2]),
                    font=font)
    
    if predictions is None:
        return img

    # iterate over all frames
    for i, pred in enumerate(zip(*predictions)):
#         print(i, pred)
        x, y = i % width, i // width
        x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

        # we can visualize multiple predictions per single frame
        for j, p in enumerate(pred):
            color = [0, 0, 0]
#             color[(j + 1) % 3] = 255
            color[0] = 255

            value = round(p * (ih - 1))
            if value != 0:
                draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=5)
    if predictions_2 is None:
        return img
    
    # iterate over all frames
    for i, pred in enumerate(zip(*predictions_2)):
#         print(i, pred)
        x, y = i % width, i // width
        x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

        # we can visualize multiple predictions per single frame
        for j, p in enumerate(pred):
            color = [0, 0, 0]
#             color[(j + 1) % 3] = 255
            color[1] = 255
            if predictions[0][i] == 1:
                color[0] = 255

            value = round(p * (ih - 1))
            if value != 0:
                draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=5)
    if predictions_3 is None:
        return img
    
    # iterate over all frames
    for i, pred in enumerate(zip(*predictions_3)):
        x, y = i % width, i // width
        x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1

        # we can visualize multiple predictions per single frame
        for j, p in enumerate(pred):
            color = [0, 0, 0]
#             color[(j + 1) % 3] = 255
            color[2] = 255
            if predictions[0][i] == 1:
                color[0] = 255
            if predictions_2[0][i] == 1:
                color[1] = 255

            value = round(p[0] * (ih - 1))
            if value != 0:
                draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=8)
    return img

def get_batches(frames, use_tqdm=True):
    reminder = 50 - len(frames) % 50
    if reminder == 50:
        reminder = 0
    frames = np.concatenate([frames[:1]] * 25 + [frames] + [frames[-1:]] * (reminder + 25), 0)

    num_batches = (len(frames) - 50) // 50  # số lượng batch (vì step size = 50)
    iterator = (
        frames[i:i + 100]
        for i in range(0, len(frames) - 50, 50)
    )

    if use_tqdm:
        return tqdm(iterator, total=num_batches, desc="[Autoshot] Processing video frames")
    else:
        return iterator

def predictions_to_scenes(predictions, threshold=0.296):
    scenes = []
    predictions = np.where(predictions > threshold, 1, 0)
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    # just fix if all predictions are 1
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)