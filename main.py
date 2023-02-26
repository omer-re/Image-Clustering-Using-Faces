import os
import shutil
import config
import face_recognition
from face_loading import loading_face
from face_encoding import get_face_encoding
from face_detection import get_face
from face_comparision import compare
import utils
from tqdm import tqdm

# utils.check_and_create_dir(config.cluster_path)
# utils.check_and_create_dir(config.sorted_path)

# get cluster count so can save new cluster without overlapping
cluster_count = sorted(os.listdir(config.cluster_path))
if len(cluster_count) > 0:
    count = cluster_count[-1] + 1
else:
    count = 0

for index, file in tqdm(enumerate(os.listdir(config.input_path)), total=len(os.listdir(config.input_path))):
    file_path = os.path.join(config.input_path, file)
    image = loading_face(file_path, face_recognition)
    face_image = get_face(image, face_recognition)
    # @Todo: Handle multiple faces in an image
    if face_image is None:
        # save the image into other category
        utils.create_dir(os.path.join(config.sorted_path, 'others'))
        shutil.copy(file_path, os.path.join(config.sorted_path, 'others', file))
        continue
    face_encoding = get_face_encoding(face_image, face_recognition)
    if face_encoding is None:
        # save the image into other category
        utils.create_dir(os.path.join(config.sorted_path, 'others'))
        shutil.copy(file_path, os.path.join(config.sorted_path, 'others', file))
        continue

    # check if there is any cluster of encodings available
    if len(os.listdir(config.cluster_path)) > 0:
        for cluster in os.listdir(config.cluster_path):
            # load cluster of encodings
            cluster_path = os.path.join(config.cluster_path, cluster)
            encoding_lists = utils.load_cluster_in_pickle(cluster_path)
            # compare face encoding with cluster
            results = compare(encoding_lists, face_encoding, face_recognition)
            is_found = False
            if len(results) > 4:
                if results.count(True) >= 3:
                    is_found = True
                else:
                    is_found = False
            else:
                if results.count(True) >= 1:
                    is_found = True
                else:
                    is_found = False
            if is_found:
                encoding_lists.append(face_encoding)
                utils.save_cluster_in_pickle(os.path.join(config.cluster_path, cluster), encoding_lists)
                shutil.copy(file_path, os.path.join(config.sorted_path, cluster.split(".")[0], file))
                break
        if not is_found:
            utils.create_dir(os.path.join(config.sorted_path, str(count)))
            shutil.copy(file_path, os.path.join(config.sorted_path, str(count), file))
            utils.save_cluster_in_pickle(os.path.join(config.cluster_path, str(count) + ".pkl"), [face_encoding])
            count = count + 1
    else:
        # make a new cluster
        os.makedirs(os.path.join(config.sorted_path, str(count)))
        shutil.copy(file_path, os.path.join(config.sorted_path, str(count), file))
        utils.save_cluster_in_pickle(os.path.join(config.cluster_path, str(count) + ".pkl"), [face_encoding])
        count = count + 1
