from skimage import io, filters, morphology, color, measure, img_as_float, feature
from sklearn import decomposition, preprocessing, linear_model
from matplotlib import cm, pyplot as plt
import matplotlib
import scipy
import pandas as pd
import numpy as np
import glob

def image_extraction(control_images=False):
    filenames = glob.glob("img/*.jpg")

    scale_positions = np.tile([slice(2300, 4300), slice(500,1400)], (len(filenames),1))
    buds_positions =  np.tile([slice(1600, 4700), slice(1320, 3000)], (len(filenames),1))
    data = pd.DataFrame()
    if control_images:
        matplotlib.use('Agg')

    for filename in filenames:

        ### read the image
        img = io.imread(filename)

        ############################## scale
        ### make binary
        binary = img[:, :, 1] < filters.threshold_otsu(img[:, :, 1])

        ### remove small object
        binary = morphology.remove_small_objects(binary, min_size=400000)

        ### remove large objects
        binary_label = measure.label(binary)
        too_big = np.bincount(binary_label.ravel()) > 900000
        too_big_mask = too_big[binary_label]
        binary[too_big_mask] = 0

        ### control image for scale
        if control_images:
            scale = np.copy(img)
            scale[np.logical_not(binary)] = scale[np.logical_not(binary)] * 0.4
            scale_filename = 'control_img/scale_' + filename.split('/')[1]
            io.imsave(scale_filename, scale)

        ### Get the coordinates of the scale
        scale_props = measure.regionprops(measure.label(binary))
        coordinates = scale_props[0].coords
        y, x = coordinates[:, 0], coordinates[:, 1]

        ### calculate conversion factors (pixels -> length or area)
        scale_area = np.sum(binary) / 10  # scale 10cm^2 --> calculated with mm
        scale_line = np.sqrt(np.sum(binary) / 10)

        ############################## buds
        ### Extract the image part with buds
        x_start = np.max(x)
        x_end = x_start + np.int((binary.shape[1] - np.max(x)) * 0.8)
        y_start = np.int(0.2 * binary.shape[0])
        y_end = np.int(0.95 * binary.shape[0])

        buds = img[y_start:y_end, x_start:x_end, 1]

        ### find a trehshold
        buds_treshold = filters.threshold_otsu(buds)
        buds_treshold = buds_treshold * 0.8

        ### histogram with treshold
        if control_images:
            _y, _x, _ = plt.hist(buds.ravel(), bins=256, color='tab:orange')
            plt.plot([buds_treshold, buds_treshold], [0, np.max(_y)], '-', color='black')
            buds_hist_filename = 'control_img/hist_buds_' + filename.split('/')[1]
            plt.savefig(buds_hist_filename)
            plt.close()

        ### make image binary
        buds_binary = buds < buds_treshold

        ### fill holes
        buds_binary = scipy.ndimage.binary_fill_holes(buds_binary)

        #### remove small objects
        buds_binary = morphology.remove_small_objects(buds_binary, min_size=2000)

        ### label the binary image
        buds_label = measure.label(buds_binary)
        if control_images:
            buds_overlay = color.label2rgb(buds_label, image=buds, bg_label=0)
            buds_overlay_filename = 'control_img/overlay_buds_' + filename.split('/')[1]
            io.imsave(buds_overlay_filename, buds_overlay)


        #### plot the data
        if control_images:
            regions = measure.regionprops(buds_label)

            plt.imshow(buds, cmap=plt.cm.gray)

            for props in regions:
                y0, x0 = props.centroid
                orientation = props.orientation
                x1 = x0 + np.cos(orientation) * 0.5 * props.minor_axis_length
                y1 = y0 - np.sin(orientation) * 0.5 * props.minor_axis_length
                x2 = x0 - np.sin(orientation) * 0.5 * props.major_axis_length
                y2 = y0 - np.cos(orientation) * 0.5 * props.major_axis_length

                plt.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
                plt.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
                plt.plot(x0, y0, '.g', markersize=15)

            buds_data_filename = 'control_img/data_buds_' + filename.split('/')[1]
            plt.savefig(buds_data_filename)
            plt.close()

        ### get properties of the objects
        props = measure.regionprops_table(buds_label,
                                          properties=('major_axis_length',
                                                      'minor_axis_length',
                                                      'area'))
        props = pd.DataFrame(props)
        props.area = props.area / scale_area
        props.iloc[:, 0:2] = props.iloc[:, 0:2] / scale_line
        props['ratio'] = props.minor_axis_length / props.major_axis_length


        img_name = filename.split('/')[1].split('.')[0]
        props['name'] = img_name.split('-')[0]
        props['location'] = img_name.split('-')[1]
        props['img_no'] = img_name.split('-')[2]
        data = data.append(props)

        print('finished with ' + filename)

    return data

def save_as_csv(data):
    data.to_csv('data/data.csv', index=False)

def load_csv():
    return pd.read_csv('data/data.csv', header=0)


def plotting(data):
    grouped_data = data.groupby(data.name)

    matplotlib.use('Agg')

    for i, d in enumerate(grouped_data):
        plt.plot(d[1].area, d[1].major_axis_length, 'o', markersize=4, label=d[0], color=cm.Set1(i))

    plt.ylabel('Length cm')
    plt.xlabel('Area cm2')
    plt.legend()
    plt.savefig('graphics/area_length.svg')
    plt.close()

def calc_pca(data):
    x = data.iloc[:,0:-3].values # remove name colums
    x = preprocessing.StandardScaler().fit_transform(x)
    pca = decomposition.PCA()
    X = pca.fit_transform(x)

    output = pd.DataFrame(X)
    output = pd.concat([output, data.name], axis=1)

    variance_explained = pca.explained_variance_ratio_

    matplotlib.use('Agg')

    species = list(np.unique(data.name))
    for i, i_species in enumerate(species):
        indices = output['name'] == i_species
        plt.plot(output.loc[indices, 0], output.loc[indices, 1], 'o',
                 color=cm.Set1(i), label=i_species)

    plt.title('PCA')
    plt.xlabel(f'PC1 ({np.round(variance_explained[0] * 100, 2)} %)')
    plt.ylabel(f'PC2 ({np.round(variance_explained[1] * 100, 2)} %)')
    plt.legend()
    plt.savefig('graphics/pca.svg')
    plt.close()

def logistic_regression(data):
    x = data.iloc[:,0:-3].values
    logreg = linear_model.LogisticRegression(solver='lbfgs')
    logreg.fit(x, data.name)
    z = logreg.predict(x)
    fit = z == data.name
    print(f'Fit: {np.sum(fit)}, out of: {len(fit)}, ({np.round(100*np.sum(fit)/len(fit),2)} %)')
    #print(logreg.predict_proba(x))

    matplotlib.use('Qt5Agg')

    grouped_data = data.groupby(data.name)
    for i, d in enumerate(grouped_data):
        plt.plot(d[1].area, d[1].major_axis_length, 'o', markersize=12, label=d[0], color=cm.Set1(i))

    species = list(np.unique(data.name))
    for i, i_species in enumerate(species):
        indices = z == i_species
        plt.plot(data.loc[indices, 'area'], data.loc[indices, 'major_axis_length'], 'o', markersize=5,
                 color=cm.Set2(i), label='pred_' + i_species)

    plt.ylabel('Length cm')
    plt.xlabel('Area cm2')
    plt.legend()
    plt.savefig('graphics/log_regression.svg')
    plt.show()


if __name__ == '__main__':
    #stats = image_extraction(control_images=True)

    #save_as_csv(stats)
    stats = load_csv()
    plotting(stats)
    logistic_regression(stats)
    calc_pca(stats)
    #print(stats)
