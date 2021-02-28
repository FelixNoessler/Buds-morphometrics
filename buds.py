from skimage import io, filters, morphology, color, measure, feature
from sklearn import decomposition, preprocessing, linear_model, ensemble, model_selection, tree
from matplotlib import patches, cm, pyplot as plt
import matplotlib
import scipy, scikit_posthocs
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import glob
import pyefd

class Extract:
    def image_extraction(self, control_images=False, extract_again=False):
        filenames = sorted(glob.glob("img/*.jpg"))

        add_info = dict()

        if not extract_again:
            remove_from_filenames = np.load('img/extracted.npy')
            for rem in remove_from_filenames:
                if rem in filenames:
                    filenames.remove(rem)

            if len(filenames) == 0:
                print('Already extracted data from all images! ')
                return

        for filename in filenames:
            ### Read the image
            img = io.imread(filename)

            ## Scale
            scale_area, scale_line, scale_coordinates = self.scale_information(img)

            ## Segementation of the buds
            buds_label, buds_treshold, buds_coordinates = self.buds_segmentation(img, scale_coordinates)

            name = 'extracted_img/' + filename.split('/')[1].split('.')[0] + '_labelled.npy'
            #np.save(name, buds_label.astype(np.int8))

            ## Plot the control image
            if control_images:
                self.plot_control_images(img, scale_coordinates, buds_coordinates, buds_treshold, buds_label, filename)


            add_info[filename.split('/')[1].split('.')[0]] = scale_area, scale_line, \
                                                             buds_coordinates[0], buds_coordinates[1], \
                                                             buds_coordinates[2], buds_coordinates[3]

            print('finished with ' + filename)


        add_info_df = pd.DataFrame(add_info).T
        add_info_df.columns = ['scale_area', 'scale_line', 'min_x', 'max_x', 'min_y', 'max_y']
        add_info_df.to_csv('extracted_img/add_info.csv', index = True)

        ### Save extracted filenames
        old_files = np.load('img/extracted.npy')
        extracted_files = np.append(old_files, filenames)
        extracted_files = np.unique(extracted_files)
        np.save('img/extracted.npy', extracted_files)
        print('Saved extracted filenames to img/extracted.npy !')


    def scale_information(self, img):
        ### make binary
        scale_binary = img[:, :, 1] < filters.threshold_otsu(img[:, :, 1])

        ### remove small object
        scale_binary = morphology.remove_small_objects(scale_binary, min_size=400000)

        ### remove large objects
        binary_label = measure.label(scale_binary)
        too_big = np.bincount(binary_label.ravel()) > 900000
        too_big_mask = too_big[binary_label]
        scale_binary[too_big_mask] = 0

        ### Get the coordinates of the scale
        scale_props = measure.regionprops(measure.label(scale_binary))
        coordinates = scale_props[0].coords
        scale_x = coordinates[:, 1]

        # min_x, max_x, min_y, max_y
        scale_coordinates = np.min(coordinates[:, 1]), np.max(coordinates[:, 1]), \
                            np.min(coordinates[:, 0]), np.max(coordinates[:, 0])

        ### calculate conversion factors (pixels -> length or area)
        scale_area = np.sum(scale_binary) / 10  # scale 10cm^2 --> calculated with mm
        scale_line = np.sqrt(np.sum(scale_binary) / 10)

        return scale_area, scale_line, scale_coordinates

    def buds_segmentation(self, img, scale_coordinates):
        ### Extract the image part with buds
        x_start = scale_coordinates[1]
        x_end = x_start + np.int((img.shape[1] - x_start) * 0.78)
        y_start = np.int(0.2 * img.shape[0])
        y_end = np.int(0.95 * img.shape[0])

        buds_coordinates = x_start, x_end, y_start, y_end

        ## Extract the part of the image with the buds
        buds = img[y_start:y_end, x_start:x_end, :]

        ## extract the blue channel
        buds_blue = buds[:, :, 2]

        ### find a trehshold
        buds_treshold = filters.threshold_otsu(buds_blue)

        ### make image binary
        buds_binary = buds_blue < buds_treshold

        ### fill holes
        buds_binary = scipy.ndimage.binary_fill_holes(buds_binary)

        #### remove small objects
        buds_binary = morphology.remove_small_objects(buds_binary, min_size=2000)

        ### label the binary image
        buds_label = measure.label(buds_binary)

        return buds_label, buds_treshold, buds_coordinates


    def plot_control_images(self, img, scale_coordinates, buds_coordinates, buds_treshold, buds_label, filename):
        """Plot the control image"""

        matplotlib.use('Qt5Agg')
        plt.figure(figsize=(10, 4))

        plt.subplot(141)
        # scale = np.copy(img)
        # scale[np.logical_not(binary)] = scale[np.logical_not(binary)] * 0.4

        plt.imshow(img)
        x_start, x_end, y_start, y_end = scale_coordinates
        plt.plot([x_start, x_end, x_end, x_start, x_start], [y_start, y_start, y_end, y_end, y_start], '-',
                 color='red')
        x_start, x_end, y_start, y_end = buds_coordinates
        plt.plot([x_start, x_end, x_end, x_start, x_start], [y_start, y_start, y_end, y_end, y_start], '-',
                 color='white')
        plt.title('Scale in red, buds in white box')

        plt.subplot(142)
        _y, _x, _ = plt.hist(img[y_start:y_end, x_start:x_end, 2].ravel(), bins=256, color='tab:orange')
        plt.plot([buds_treshold, buds_treshold], [0, np.max(_y)], '--', color='black', label='threshold')
        # plt.legend()
        plt.title('Tresholding of the buds')

        plt.subplot(143)
        plt.title('Axis lengths of the buds')
        regions = measure.regionprops(buds_label)

        buds_overlay = color.label2rgb(buds_label, image=img[y_start:y_end, x_start:x_end, 2], bg_label=0)
        plt.imshow(buds_overlay)

        for props in regions:
            y0, x0 = props.centroid
            orientation = props.orientation

            ## minor axis
            x_min0 = x0 - np.cos(orientation) * 0.5 * props.minor_axis_length
            y_min0 = y0 + np.sin(orientation) * 0.5 * props.minor_axis_length
            x_min1 = x0 + np.cos(orientation) * 0.5 * props.minor_axis_length
            y_min1 = y0 - np.sin(orientation) * 0.5 * props.minor_axis_length
            plt.plot([x_min0, x_min1], [y_min0, y_min1], '-r', linewidth=1)

            ## major axis
            x_maj0 = x0 + np.sin(orientation) * 0.5 * props.major_axis_length
            y_maj0 = y0 + np.cos(orientation) * 0.5 * props.major_axis_length
            x_maj1 = x0 - np.sin(orientation) * 0.5 * props.major_axis_length
            y_maj1 = y0 - np.cos(orientation) * 0.5 * props.major_axis_length
            plt.plot([x_maj0, x_maj1], [y_maj0, y_maj1], '-r', linewidth=1)

        plt.subplot(144)
        bud = img[y_start:y_end, x_start:x_end, 2][regions[1].slice]
        bud_bin = buds_label[regions[1].slice]
        overlay = color.label2rgb(bud_bin, image=bud, bg_label=0)
        plt.imshow(overlay)
        plt.title('Enlarged first bud ')

        plt.tight_layout()
        control_filename = 'control_img/' + filename.split('/')[1]
        plt.savefig(control_filename)
        plt.close()


class Info:
    def collect_information(self, color=True, graycomatrix=True, fourier=True):
        filenames = sorted(glob.glob("img/*.jpg"))

        dict_of_coefficients = dict()
        data = pd.DataFrame()
        self.add_info = pd.read_csv('extracted_img/add_info.csv', header=0)

        for filename in filenames:
            ### Read the image
            self.img = io.imread(filename)

            name = 'extracted_img/' + filename.split('/')[1].split('.')[0] + '_labelled.npy'
            self.buds_label = np.load(name)

            self.f = filename

            properties = self.extract_information()

            if color:
                hsv, int = self.extract_color_props()
                properties = {**properties, **hsv, **int}

            if graycomatrix:
                gray = self.extract_greycomatrix()
                properties = {**properties, **gray}

            if fourier:
                coefficients = self.extract_fourier_coefficients()
                self.contour_controlplot()
                dict_of_coefficients = {**dict_of_coefficients, **coefficients}

            ### convert to dataframe
            props_df = pd.DataFrame(properties)

            ### save also metadata from filename
            img_name = filename.split('/')[1].split('.')[0]
            props_df['name'] = img_name.split('-')[0]
            props_df['location'] = img_name.split('-')[1]
            props_df['img_no'] = img_name.split('-')[2]

            ### Append to big table of buds from all images
            data = data.append(props_df)
            print('finished with ' + filename)


        data.to_csv('data/data.csv', index=False)
        print('Saved extracted data!')

        if fourier:
            np.savez('data/EF_coeffcicients.npz', **dict_of_coefficients)
            print('Saved fourier coefficients!')



    def extract_information(self):
        ## get properties of the objects
        props = measure.regionprops_table(self.buds_label,
                                          properties=('label',
                                                      'major_axis_length',
                                                      'minor_axis_length',
                                                      'area',
                                                      'perimeter'))

        mask = self.add_info.iloc[:, 0] == self.f.split('/')[1].split('.')[0]

        # scaling
        props['area'] = props['area'] / self.add_info[mask]['scale_area'].values
        props['major_axis_length'] = props['major_axis_length'] / self.add_info[mask]['scale_line'].values
        props['minor_axis_length'] = props['minor_axis_length'] / self.add_info[mask]['scale_line'].values
        props['perimeter'] = props['perimeter'] / self.add_info[mask]['scale_line'].values

        ## calculate ratio of major/minor axis length and roundness
        add = dict(ratio=props['minor_axis_length'] / props['major_axis_length'],
                   roundness=4 * np.pi * props['area'] / props['perimeter'] ** 2)

        properties = {**props, **add}

        return properties


    def extract_color_props(self):
        ## Extract information from HSV color model
        mask = self.add_info.iloc[:, 0] == self.f.split('/')[1].split('.')[0]
        x_start, x_end, y_start, y_end = self.add_info[mask].iloc[0, 3:7].astype(int)

        hsv = color.rgb2hsv(self.img[y_start:y_end, x_start:x_end, :])

        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        hue_median, hue_mean, hue_std = list(), list(), list()
        sat_median, sat_mean, sat_std = list(), list(), list()
        val_median, val_mean, val_std = list(), list(), list()

        labels = np.arange(1, np.max(self.buds_label) + 1)
        for l in labels:
            mask = self.buds_label == l

            # hue
            hue_l = hue[mask]
            hue_median.append(np.median(hue_l))
            hue_mean.append(np.mean(hue_l))
            hue_std.append(np.std(hue_l))

            # saturation
            sat_l = saturation[mask]
            sat_median.append(np.median(sat_l))
            sat_mean.append(np.mean(sat_l))
            sat_std.append(np.std(sat_l))

            # value
            val_l = value[mask]
            val_median.append(np.median(val_l))
            val_mean.append(np.mean(val_l))
            val_std.append(np.std(val_l))

        hsv_data = dict(hue_mean=hue_mean, hue_median=hue_median, hue_std=hue_std,
                        sat_median=sat_median, sat_mean=sat_mean, sat_std=sat_std,
                        val_median=val_median, val_mean=val_mean, val_std=val_std)

        ## Extract information about the intensity distribution (Blue channel)
        regions = measure.regionprops(self.buds_label, intensity_image=self.img[y_start:y_end, x_start:x_end, 2])
        int_med, int_std, int_skew = [], [], []
        for i in range(0, np.max(self.buds_label)):
            intensity = regions[i].intensity_image.ravel()
            intensity = intensity[intensity != 0]
            int_med.append(np.median(intensity))
            int_std.append(np.std(intensity))
            int_skew.append(scipy.stats.skew(intensity))

        intensity_data = dict(int_med=int_med, int_std=int_std, int_skew=int_skew)

        return hsv_data, intensity_data


    def extract_greycomatrix(self):
        ## Gray level co-occurrence matrix
        contrast, homogeneity, correlation = list(), list(), list()
        mask = self.add_info.iloc[:, 0] == self.f.split('/')[1].split('.')[0]
        x_start, x_end, y_start, y_end = self.add_info[mask].iloc[0, 3:7].astype(int)
        buds = self.img[y_start:y_end, x_start:x_end, 2]
        for region in measure.regionprops(self.buds_label):
            min_row, min_col, max_row, max_col = region.bbox
            box = buds[min_row:max_row, min_col:max_col]
            matrix = feature.greycomatrix(box, distances=[1], angles=[0], symmetric=True, normed=True)
            con = feature.greycoprops(matrix, 'contrast')
            contrast.append(float(con))
            h = feature.greycoprops(matrix, 'homogeneity')
            homogeneity.append(float(h))
            cor = feature.greycoprops(matrix, 'correlation')
            correlation.append(float(cor))

        gray_comatrix_data = dict(contrast=contrast, homogeneity=homogeneity, correlation=correlation)

        return gray_comatrix_data


    def extract_fourier_coefficients(self):
        ## Coefficients of the Fourier series (an, bn, cn and dn)
        regions = measure.regionprops(self.buds_label)
        coefficients = dict()

        self.control_img_old_contour, self.control_img_final_contour = list(), list()

        for r in regions:
            n = self.f.split('/')[1].split('.')[0] + f'-{r.label}'
            a = r.filled_image
            z = np.zeros((a.shape[0] + 2, a.shape[1] + 2))
            z[1:-1, 1:-1] = a

            contour = measure.find_contours(z, 0.5)[0]

            # Ellipse of old contour
            ellipse = measure.EllipseModel()
            ellipse.estimate(contour)
            ellipse_cord = ellipse.predict_xy(np.linspace(0, 2 * np.pi, 25))

            if ellipse.params[2] > ellipse.params[3]:
                angle = 90 - ellipse.params[4] * 180 / np.pi
            else:
                angle = 180 - ellipse.params[4] * 180 / np.pi

            new_contour = self.rotate(contour, angle)



            # Rotate flipped buds
            lower_mask = new_contour[:, 1] < np.quantile(new_contour[:, 1], 0.4)
            low_dist = np.max(new_contour[lower_mask, 0]) - np.min(new_contour[lower_mask, 0])

            higher_mask = new_contour[:, 1] > np.quantile(new_contour[:, 1], 0.6)
            high_dist = np.max(new_contour[higher_mask, 0]) - np.min(new_contour[higher_mask, 0])


            if low_dist < high_dist:
                flipped_contour = self.rotate(new_contour, 180)
            else:
                flipped_contour = new_contour



            max = np.max(flipped_contour[:, 1])
            max_pos = np.where(flipped_contour[:, 1] == max)[0][0]

            start = flipped_contour[max_pos:,:]
            end = flipped_contour[1:max_pos+1,:]

            final_contour = np.concatenate((start, end))

            new_coef = pyefd.elliptic_fourier_descriptors(
                np.squeeze(final_contour), order=50, normalize=False)

            coefficients[n] = new_coef

            self.control_img_old_contour.append(contour)
            self.control_img_final_contour.append(final_contour)

        return coefficients

    def cart2pol(self, x, y):
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return theta, rho

    def pol2cart(self, theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    def rotate(self, contour, angle):
        M = measure.moments_coords(contour)
        cx = M[1, 0] / M[0, 0]
        cy = M[0, 1] / M[0, 0]
        contour_norm = contour - [cx, cy]

        thetas, rhos = self.cart2pol(contour_norm[:, 0], contour_norm[:, 1])

        thetas = np.rad2deg(thetas)
        thetas = (thetas + angle) % 360
        thetas = np.deg2rad(thetas)

        xs, ys = self.pol2cart(thetas, rhos)

        contour_norm[:, 0] = xs
        contour_norm[:, 1] = ys

        contour_rotated = contour_norm + [cx, cy]

        return contour_rotated

    def contour_controlplot(self):
        plt.figure(figsize=(18, 9))
        for i, (old_contour, final_contour) in enumerate(zip(self.control_img_old_contour,
                                                             self.control_img_final_contour)):
            plt.subplot(2, int((len(self.control_img_old_contour) + 1) / 2), i + 1)
            plt.plot(final_contour[:, 0], final_contour[:, 1], 'o', markersize=1)
            plt.plot(old_contour[:, 0], old_contour[:, 1], 'o', markersize=1)
            plt.gca().set_aspect('equal', adjustable='box')

        plt.tight_layout()
        img_name = 'control_img/contour/' + self.f.split('/')[1].split('.')[0] + '.png'
        plt.savefig(img_name)
        plt.close()


class Analyse:
    def __init__(self):
        self.coef = self.load_coefficients()
        self.data = pd.read_csv('data/data.csv', header=0)

    def load_coefficients(self):
        file = np.load('data/EF_coeffcicients.npz')
        names = file.files
        all_data = np.empty((0, file[names[0]].shape[0] * file[names[0]].shape[1]))

        for n in names:
            f = file[n]
            #f = pyefd.normalize_efd(f)
            f = f.reshape((1, f.shape[0] * f.shape[1]))
            all_data = np.append(all_data, f, axis=0)

        return all_data


    def data_prep(self, combine_coef=False, only_coef=False):
        pd.set_option('display.max_columns', 10)
        #print(data.iloc[:, 1:-3].head())

        if combine_coef:
            x = self.data.iloc[:, 1:-3].values
            x = np.concatenate([x, self.coef], axis=1)
        elif only_coef:
            x = self.coef
        else:
            x = self.data.iloc[:, 1:-3].values

        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)
        y = self.data.name
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)
        return x_train, x_test, y_train, y_test


    def area_length_plot(self):
        grouped_data = self.data.groupby(self.data.name)

        for i, d in enumerate(grouped_data):
            plt.plot(d[1].area, d[1].major_axis_length, 'o',
                     markersize=2, label=d[0], color=cm.tab10(i))

        plt.ylabel('Length cm')
        plt.xlabel(r'Area cm$^2$')
        plt.legend(markerscale=3.0)
        plt.savefig('graphics/area_length.svg')
        plt.show()
        plt.close()

    def spider_plot(self):
        species = np.unique(self.data.name)
        categories = self.data.iloc[:, 1:-3].columns
        mean_values = self.data.iloc[:, 1:-2].groupby('name').mean().values

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


    def make_boxplot(self):
        data = self.data
        cols = list(data.iloc[:, 1:-3])

        code = pd.Categorical(data.name).codes
        data_splitted = [data[cols].values[code == l] for l in np.unique(code)]
        f, p = scipy.stats.f_oneway(*data_splitted)
        print(f'Anova p = {p}')



        splitted = [data[cols[0]].values[code == l] for l in np.unique(code)]
        scipy.stats.kruskal(*splitted)

        res = scikit_posthocs.posthoc_mannwhitney(data, val_col=cols[0], group_col='name', p_adjust = 'holm')

        x_label = [la.split()[0][0] + la.split()[1][0] for la in np.unique(data.name)]

        res.columns, res.index = x_label, x_label
        cmap = ['1', '#fb6a4a', '#08306b', '#4292c6', '#c6dbef']
        heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        scikit_posthocs.sign_plot(res, **heatmap_args)
        plt.savefig('test.svg')


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

        [ax_tmp.set_xticklabels(x_label) for ax_tmp in axs.reshape(-1)]
        [ax_tmp.set_xlabel('Species') for ax_tmp in axs.reshape(-1)]

        plt.savefig('graphics/boxplot.svg')
        plt.show()


    def pca(self):

        textsize = 12

        # remove name colums
        x = self.data.iloc[:,1:-3].values

        x = preprocessing.StandardScaler().fit_transform(x)
        pca = decomposition.PCA()
        X = pca.fit_transform(x)

        ndims = np.shape(x)[1]

        output = pd.DataFrame(X)
        output = pd.concat([output, self.data.name], axis=1)


        variance_explained = pca.explained_variance_ratio_

        species = list(np.unique(self.data.name))
        count_categories = self.data.groupby('name').count().iloc[:, 0].values

        fig, ax = plt.subplots(nrows=1, ncols=1)

        for i, i_species in enumerate(species):
            indices = output['name'] == i_species

            italic_species = '$ \it{ '  + i_species.split()[0]  + '}$ ' + '$ \it{' + i_species.split()[1] + '}$'
            plot_label = italic_species + f', n={count_categories[i]}'

            x_pca = output.loc[indices, 0]
            y_pca = output.loc[indices, 1]

            ######
            nstd = 2
            cov = np.cov(x_pca, y_pca)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * nstd * np.sqrt(vals)
            ell = patches.Ellipse(xy=(np.mean(x_pca), np.mean(y_pca)),
                          width=w, height=h, angle=theta,
                          edgecolor=cm.tab10(i), fc=cm.tab10(i), lw = 1, alpha=0.1)
            ax.add_patch(ell)
            ######

            ax.plot(x_pca, y_pca, 'o', color=cm.tab10(i), label=plot_label, markersize=2.4)

        self.h, self.l = ax.get_legend_handles_labels()



        plt.title(f'PCA, input dims: {ndims}', size=textsize*1.2)
        plt.xlabel(f'PC1 ({np.round(variance_explained[0] * 100, 2)} %)', size=textsize)
        plt.ylabel(f'PC2 ({np.round(variance_explained[1] * 100, 2)} %)', size=textsize)
        plt.legend(loc='center', bbox_to_anchor=(1.35, 0.5), markerscale=3.0, prop={'size': textsize})
        plt.savefig('graphics/pca.svg', bbox_inches="tight")
        plt.show()
        plt.close()

    def pca_fourier(self):
        textsize = 12

        pca = decomposition.PCA()
        X = pca.fit_transform(self.coef)

        variance_explained = pca.explained_variance_ratio_

        color = [np.where(np.unique(self.data.name) == self.data.name[i])[0][0] for i in range(len(self.data.name))]

        plt.figure(figsize=(15,7))
        plt.subplot2grid((1, 4), (0, 0), colspan=2)
        for i in range(len(self.data)):
            plt.plot(X[i, 0], X[i,1], 'o', markersize=2, color=cm.tab10(color[i]))

        plt.title(f'PCA of EFA data', size=textsize*1.2)
        plt.xlabel(f'PC1 ({np.round(variance_explained[0] * 100, 2)} %)', size=textsize)
        plt.ylabel(f'PC2 ({np.round(variance_explained[1] * 100, 2)} %)', size=textsize)
        plt.legend(self.h, self.l, markerscale=3)

        ## axis
        first = X[:,0]

        selection = first  <= np.quantile(first, 0.1)
        coef_sel = self.coef[selection, :]
        m = np.mean(coef_sel, axis=0)
        inv = pyefd.reconstruct_contour(m.reshape(int(m.shape[0]/4), 4))


        selection2 = first >= np.quantile(first, 0.9)
        coef_sel2 = self.coef[selection2, :]
        m2 = np.mean(coef_sel2, axis=0)
        inv2 = pyefd.reconstruct_contour(m2.reshape(int(m.shape[0]/4), 4))

        plt.subplot2grid((1, 4), (0, 2))
        plt.plot(inv[:, 0], inv[:, 1], label='low 10 %')
        plt.plot(inv2[:, 0], inv2[:, 1], label='high 10 %')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Variation along first axis')
        plt.legend()

        ### 2. axis
        second = X[:,1]

        selection = second  <= np.quantile(second, 0.1)
        coef_sel = self.coef[selection, :]
        m = np.mean(coef_sel, axis=0)
        inv = pyefd.reconstruct_contour(m.reshape(int(m.shape[0]/4), 4))


        selection2 = second >= np.quantile(second, 0.9)
        coef_sel2 = self.coef[selection2, :]
        m2 = np.mean(coef_sel2, axis=0)
        inv2 = pyefd.reconstruct_contour(m2.reshape(int(m.shape[0]/4), 4))

        plt.subplot2grid((1, 4), (0, 3))
        plt.plot(inv[:, 0], inv[:, 1], label='low values second axis')
        plt.plot(inv2[:, 0], inv2[:, 1], label='high values second axis')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Variation along second axis')
        #plt.legend()

        plt.tight_layout()

        plt.savefig('graphics/pca_efa.svg')
        plt.show()


    def logistic_regression(self):
        x_train, x_test, y_train, y_test = self.data_prep()

        logreg = linear_model.LogisticRegression(max_iter=200)
        logreg.fit(x_train, y_train)
        z = logreg.predict(x_train)
        fit = z == y_train
        print(f'Logistic Regression - Train data fit: '
              f'{np.sum(fit)}, out of: {len(fit)} ({np.round(100*np.sum(fit)/len(fit),2)} %)')

        z = logreg.predict(x_test)
        fit = z == y_test
        print(f'Logistic Regression - Test data fit: '
              f'{np.sum(fit)}, out of: {len(fit)} ({np.round(100 * np.sum(fit) / len(fit), 2)} %)')


    def random_forest(self):
        x_train, x_test, y_train, y_test = self.data_prep()

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

    def decision_tree(self):
        x_train, x_test, y_train, y_test = self.data_prep()
        tree1 = tree.DecisionTreeClassifier(max_depth=10)

        tree1.fit(x_train, y_train)
        print(f'Decision tree - Train score: {np.round(100*tree1.score(x_train, y_train),2)} %')
        print(f'Decision tree - Test score: {np.round(100*tree1.score(x_test, y_test),2)} %')

        plt.figure(figsize=(40,30))
        tree.plot_tree(tree1,filled=True,
                       feature_names=self.data.iloc[:, 1:-3].columns,
                       class_names=np.unique(self.data.name))
        plt.tight_layout()
        plt.savefig('graphics/descision_tree.png')
        plt.close()

        n = x_train.shape[1]
        plt.barh(range(n), tree1.feature_importances_, align='center')
        plt.yticks(np.arange(n), self.data.iloc[:, 1:-3].columns)
        plt.xlabel('Importance of feature')
        plt.ylabel('feature')
        plt.tight_layout()
        plt.savefig('graphics/coefs_decision_tree.svg')


if __name__ == '__main__':
    #e = Extract()
    #e.image_extraction(control_images=False, extract_again=True)
    #i = Info()
    #i.collect_information()
    a = Analyse()
    #a.pca()
    #a.pca_fourier()
    a.logistic_regression()
    a.random_forest()
    a.decision_tree()
