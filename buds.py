from skimage import io, filters, morphology, color, measure, img_as_ubyte, feature
from sklearn import decomposition, preprocessing, linear_model, ensemble, model_selection, tree
from matplotlib import cm, pyplot as plt
import matplotlib
import scipy
import pandas as pd
import seaborn
import numpy as np
import glob


def image_extraction(control_images=False, extract_again=False):
    filenames = sorted(glob.glob("img/*.jpg"))
    data = pd.DataFrame()

    if not extract_again:
        remove_from_filenames = np.load('img/extracted.npy')
        for rem in remove_from_filenames:
            if rem in filenames:
                filenames.remove(rem)

        if len(filenames) == 0:
            print('Already extracted data from all images! ')
            return


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
        x_end = x_start + np.int((binary.shape[1] - np.max(x)) * 0.78)
        y_start = np.int(0.2 * binary.shape[0])
        y_end = np.int(0.95 * binary.shape[0])

        buds = img[y_start:y_end, x_start:x_end, 1]

        ### find a trehshold
        buds_treshold = filters.threshold_otsu(buds) * 0.95

        ### make image binary
        buds_binary = buds < buds_treshold

        ### fill holes
        buds_binary = scipy.ndimage.binary_fill_holes(buds_binary)

        #### remove small objects
        buds_binary = morphology.remove_small_objects(buds_binary, min_size=2000)

        ### label the binary image
        buds_label = measure.label(buds_binary)


        ### grey-level co-occurrence matrix
        regions = measure.regionprops(buds_label, buds)
        diss, homo, corr = np.array([]),np.array([]),np.array([])
        for i in range(0, np.max(buds_label)):
            matrix = feature.texture.greycomatrix(regions[i].intensity_image, distances=[5], angles=[0])
            diss = np.append(diss, feature.texture.greycoprops(matrix, 'dissimilarity'))
            homo = np.append(homo, feature.texture.greycoprops(matrix, 'homogeneity'))
            corr = np.append(corr, feature.texture.greycoprops(matrix, 'correlation'))

        #### plot the control image
        if control_images:
            matplotlib.use('Agg')
            plt.figure(figsize=(10,4))

            plt.subplot(131)
            scale = np.copy(img)
            scale[np.logical_not(binary)] = scale[np.logical_not(binary)] * 0.4
            plt.imshow(scale)
            plt.plot([x_start, x_end, x_end, x_start, x_start], [y_start, y_start, y_end, y_end, y_start], '-', color='white')
            plt.title('Scale in red, buds in white box')

            plt.subplot(132)
            _y, _x, _ = plt.hist(buds.ravel(), bins=256, color='tab:orange')
            plt.plot([buds_treshold, buds_treshold], [0, np.max(_y)], '-', color='black', label='threshold')
            plt.legend()
            plt.title('Tresholding of the buds')

            plt.subplot(133)
            plt.title('Axis lengths of the buds')
            regions = measure.regionprops(buds_label)

            buds_overlay = color.label2rgb(buds_label, image=buds, bg_label=0)
            plt.imshow(buds_overlay)

            for props in regions:
                y0, x0 = props.centroid
                orientation = props.orientation

                ## minor axis
                x_min0 = x0 - np.cos(orientation) * 0.5 * props.minor_axis_length
                y_min0 = y0 + np.sin(orientation) * 0.5 * props.minor_axis_length
                x_min1 = x0 + np.cos(orientation) * 0.5 * props.minor_axis_length
                y_min1 = y0 - np.sin(orientation) * 0.5 * props.minor_axis_length
                plt.plot((x_min0, x_min1), (y_min0, y_min1), '-r', linewidth=1)

                ## major axis
                x_maj0 = x0 + np.sin(orientation) * 0.5 * props.major_axis_length
                y_maj0 = y0 + np.cos(orientation) * 0.5 * props.major_axis_length
                x_maj1 = x0 - np.sin(orientation) * 0.5 * props.major_axis_length
                y_maj1 = y0 - np.cos(orientation) * 0.5 * props.major_axis_length
                plt.plot([x_maj0, x_maj1], [y_maj0, y_maj1], '-r', linewidth=1)

            control_filename = 'control_img/' + filename.split('/')[1]
            plt.savefig(control_filename)
            plt.close()

        ### get properties of the objects
        props = measure.regionprops_table(buds_label,
                                          properties=('label',
                                                      'major_axis_length',
                                                      'minor_axis_length',
                                                      'area',
                                                      'perimeter'))
        props = pd.DataFrame(props)
        props.area = props.area / scale_area
        props.major_axis_length = props.major_axis_length / scale_line
        props.minor_axis_length = props.minor_axis_length / scale_line
        props.perimeter = props.perimeter / scale_line
        props['ratio'] = props.minor_axis_length / props.major_axis_length
        props['roundness'] = 4 * np.pi * props.area / props.perimeter**2

        props['homogeneity'] = homo
        props['dissimilarity']  = diss
        props['correlation'] = corr

        img_name = filename.split('/')[1].split('.')[0]
        props['name'] = img_name.split('-')[0]
        props['location'] = img_name.split('-')[1]
        props['img_no'] = img_name.split('-')[2]
        data = data.append(props)

        print('finished with ' + filename)


    ### Save extracted filenames
    old_files = np.load('img/extracted.npy')
    extracted_files = np.append(old_files, filenames)
    extracted_files = np.unique(extracted_files)
    np.save('img/extracted.npy', extracted_files)
    print('Saved extracted filenames to img/extracted.npy !')

    return data


def save_as_csv(data, overwrite=False):
    if overwrite:
        data.to_csv('data/data.csv', index=False)
    else:
        old_data = pd.read_csv('data/data.csv', header=0)
        new_data = old_data.append(data)
        new_data.to_csv('data/data.csv', index=False)

def load_csv():
    return pd.read_csv('data/data.csv', header=0)



def plotting(data):
    grouped_data = data.groupby(data.name)

    matplotlib.use('Agg')

    for i, d in enumerate(grouped_data):
        plt.plot(d[1].area, d[1].major_axis_length, 'o',
                 markersize=2, label=d[0], color=cm.tab10(i))

    plt.ylabel('Length cm')
    plt.xlabel('Area cm2')
    plt.legend()
    plt.savefig('graphics/area_length.svg')
    plt.close()


def calc_pca(data):
    x = data.iloc[:,1:-3].values # remove name colums
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
                 color=cm.tab10(i), label=i_species, markersize=2)

    plt.title('PCA')
    plt.xlabel(f'PC1 ({np.round(variance_explained[0] * 100, 2)} %)')
    plt.ylabel(f'PC2 ({np.round(variance_explained[1] * 100, 2)} %)')
    plt.legend()
    plt.savefig('graphics/pca.svg')
    plt.close()


def logistic_regression(data):
    x_train, x_test, y_train, y_test = data_prep(data)

    logreg = linear_model.LogisticRegression()
    logreg.fit(x_train, y_train)
    z = logreg.predict(x_train)
    fit = z == y_train
    print(f'Logistic Regression - Train data fit: '
          f'{np.sum(fit)}, out of: {len(fit)} ({np.round(100*np.sum(fit)/len(fit),2)} %)')
    #print(logreg.predict_proba(x))


    z = logreg.predict(x_test)
    fit = z == y_test
    print(f'Logistic Regression - Test data fit: '
          f'{np.sum(fit)}, out of: {len(fit)} ({np.round(100 * np.sum(fit) / len(fit), 2)} %)')

    return
    matplotlib.use('Qt5Agg')
    z = logreg.predict(x)
    grouped_data = data.groupby(data.name)
    for i, d in enumerate(grouped_data):
        plt.plot(d[1].area, d[1].major_axis_length, 'o', markersize=4, label=d[0], color=cm.Set1(i))

    species = list(np.unique(data.name))
    for i, i_species in enumerate(species):
        indices = z == i_species
        plt.plot(data.loc[indices, 'area'], data.loc[indices, 'major_axis_length'], 'o', markersize=2,
                 color=cm.Set2(i), label='pred_' + i_species)

    plt.ylabel('Length cm')
    plt.xlabel('Area cm2')
    plt.legend()
    plt.savefig('graphics/log_regression.svg')
    plt.show()


def random_forest(data):
    x_train, x_test, y_train, y_test = data_prep(data)

    clf = ensemble.RandomForestClassifier(n_estimators=15)
    clf.fit(x_train, y_train)

    # train data
    z = clf.predict(x_train)
    fit = z == y_train
    print(f'Random Forest - Train data fit: {np.sum(fit)} out of {len(fit)} ({np.round(100 * np.sum(fit) / len(fit), 2)} %)')

    # test data
    z = clf.predict(x_test)
    fit = z == y_test
    print(f'Random Forest - Test data fit: {np.sum(fit)} out of {len(fit)} ({np.round(100 * np.sum(fit) / len(fit), 2)} %)')

def decision_tree(data):
    x_train, x_test, y_train, y_test = data_prep(data)

    tree1 = tree.DecisionTreeClassifier(max_depth=9)

    tree1.fit(x_train, y_train)
    print(f'Decision tree - Train score: {np.round(100*tree1.score(x_train, y_train),2)}')
    print(f'Decision tree - Test score: {np.round(100*tree1.score(x_test, y_test),2)}')


    tree.plot_tree(tree1)
    plt.show()

def spider_plot(data):
    species = np.unique(data.name)
    categories = data.iloc[:, 1:-3].columns
    mean_values = data.iloc[:, 1:-2].groupby('name').mean().values

    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    #plt.figure(figsize=(9,5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8,  1], ["0", "0.2","0.4", "0.6", "0.8", "1"], color="grey", size=7)
    plt.ylim(0, 1)

    for i in range(N):
        mean_values[:, i] /= np.max(mean_values[:, i])

    for i in range(len(mean_values)):
        values = mean_values[i,:]
        values = np.append(values, values[:1])
        plt.plot(angles, values, linewidth=1.5, linestyle='solid', label=species[i])

    plt.legend(loc='center', bbox_to_anchor=(1.4, 0.5))
    plt.savefig("graphics/spider_plot.svg", bbox_inches="tight")
    plt.show()

def data_prep(data):
    pd.set_option('display.max_columns', 10)
    #print(data.iloc[:, 1:-3].head())
    x = data.iloc[:, 1:-3].values
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    y = data.name
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test

def make_boxplot(data):

    cols = list(data.iloc[:, 1:-3])

    code = pd.Categorical(data.name).codes
    data_splitted = [data[cols].values[code == l] for l in np.unique(code)]
    f, p = scipy.stats.f_oneway(*data_splitted)
    print(f'Anova p = {p}')

    area_splitted = [data.area.values[code == l] for l in np.unique(code)]
    f, p = scipy.stats.kruskal(*area_splitted)
    print(f'Kruskal p = {p}')

    fig, axs = plt.subplots(1, len(cols),  figsize=(18, 5), sharey=False)

    prop_dic = {'patch_artist': True,
                'boxprops': dict(color='k', facecolor='tab:blue'),
                'capprops': dict(color='k'),
                'flierprops': dict(color='k', markeredgecolor='k', markerfacecolor='k', markersize=3),
                'medianprops': dict(color='k'),
                'whiskerprops': dict(color='k')}

    data.boxplot(column=cols, by='name', ax=axs, **prop_dic)
    fig.suptitle('')
    x_label = [la.split()[0][0] + la.split()[1][0] for la in np.unique(data.name)]
    [ax_tmp.set_xticklabels(x_label) for ax_tmp in axs.reshape(-1)]
    [ax_tmp.set_xlabel('Species') for ax_tmp in axs.reshape(-1)]

    plt.savefig('graphics/boxplot.svg')
    plt.show()



if __name__ == '__main__':
    #stats = image_extraction(control_images=True, extract_again=True)
    #save_as_csv(stats, overwrite=True)
    stats = load_csv()
    #spider_plot(stats)
    make_boxplot(stats)
    #plotting(stats)
    #logistic_regression(stats)
    #calc_pca(stats)
    #random_forest(stats)
    #decision_tree(stats)
