import numpy as np
import neurospyke as ns
from sklearn.decomposition import PCA

def reduce_IFR_dimensions(IFR, n_components, fixed_field):
    NPD = np.zeros([np.shape(IFR)[0],np.shape(IFR)[1], n_components, np.shape(IFR)[3], np.shape(IFR)[4]])

    if fixed_field == 'areas':
        for areas_idx in np.arange(np.shape(NPD)[1]):
            IFR_trials = IFR[:, areas_idx, :, :, :]
            IFR_mean_trials = np.mean(IFR_trials, axis=0)
            IFR_mean_trials = np.mean(IFR_mean_trials, axis=1)

            areas_PCA = PCA(n_components)
            areas_PCA.fit(np.transpose(IFR_mean_trials))

            for conditions_idx in np.arange(np.shape(NPD)[0]):
                for trial_idx in np.arange(np.shape(NPD)[3]):
                    NPD[conditions_idx, areas_idx, :, trial_idx, :] = np.transpose(
                        areas_PCA.transform(
                            np.transpose(IFR_trials[conditions_idx, :, trial_idx, :])
                        )
                    )
    elif fixed_field == 'conditions':
        for conditions_idx in np.arange(np.shape(NPD)[0]):
            IFR_trials = IFR[conditions_idx, :, :, :, :]
            IFR_mean_trials = np.mean(IFR_trials, axis=0)
            IFR_mean_trials = np.mean(IFR_mean_trials, axis=1)

            conditions_PCA = PCA(n_components)
            conditions_PCA.fit(np.transpose(IFR_mean_trials))

            for areas_idx in np.arange(np.shape(NPD)[1]):
                for trial_idx in np.arange(np.shape(NPD)[3]):
                    NPD[conditions_idx, areas_idx, :, trial_idx, :] = np.transpose(
                        conditions_PCA.transform(
                            np.transpose(IFR_trials[areas_idx, :, trial_idx, :])
                        )
                    )
    else:
        raise ValueError("'" + fixed_field + "' cannot be fixed.")

    return NPD

def plot_neural_population_dynamics(NPD, n_components, fixed_field, title='Neural Population Dynamics', figsize=[12, 6], dpi=100):
    trial_range = np.arange(np.shape(NPD)[4]).astype(np.int_)

    fig = ns.visualization.pyplot.figure(dpi=dpi, figsize=figsize)
    if n_components == 3:
        axs = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]
    else:
        axs = [fig.add_subplot(121), fig.add_subplot(122)]
    
    if fixed_field == 'areas':
        axs_titles = ['RFA', 'S1']
        handles = [None, None]

        colors = ['#1F7508', '#FF0000']
        colors_markers = [['#194706', '#12230B'], ['#800606', '#780037']]

        for condition_idx in np.arange(np.shape(NPD)[0]):
            color = colors[condition_idx]
            for area_idx in np.arange(np.shape(NPD)[1]):
                ax = axs[area_idx]
                for trial_idx in np.arange(np.shape(NPD)[3]):
                    if n_components == 3:
                        handles[condition_idx], = ax.plot(
                            NPD[condition_idx, area_idx, 0, trial_idx, trial_range],
                            NPD[condition_idx, area_idx, 1, trial_idx, trial_range],
                            NPD[condition_idx, area_idx, 2, trial_idx, trial_range],
                            color=color,
                            linewidth=0.5
                        )
                        ax.scatter(
                            NPD[condition_idx, area_idx, 0, trial_idx, trial_range[0]], 
                            NPD[condition_idx, area_idx, 1, trial_idx, trial_range[0]], 
                            NPD[condition_idx, area_idx, 2, trial_idx, trial_range[0]],
                            color=colors_markers[condition_idx][0],
                            s=8,
                            marker='*',

                        )
                        ax.scatter(
                            NPD[condition_idx, area_idx, 0, trial_idx, trial_range[-1]], 
                            NPD[condition_idx, area_idx, 1, trial_idx, trial_range[-1]], 
                            NPD[condition_idx, area_idx, 2, trial_idx, trial_range[-1]],
                            color=colors_markers[condition_idx][1],
                            s=8,
                            marker='<'
                        )
                    else:    
                        handles[condition_idx], = ax.plot(
                            NPD[condition_idx, area_idx, 0, trial_idx, trial_range],
                            NPD[condition_idx, area_idx, 1, trial_idx, trial_range],
                            color=color,
                            linewidth=0.5
                        )
                        ax.scatter(
                            NPD[condition_idx, area_idx, 0, trial_idx, trial_range[0]], 
                            NPD[condition_idx, area_idx, 1, trial_idx, trial_range[0]],
                            color=colors_markers[condition_idx][0],
                            s=8,
                            marker='*',

                        )
                        ax.scatter(
                            NPD[condition_idx, area_idx, 0, trial_idx, trial_range[-1]], 
                            NPD[condition_idx, area_idx, 1, trial_idx, trial_range[-1]],
                            color=colors_markers[condition_idx][1],
                            s=8,
                            marker='<'
                        )
            
                ax.set_title(axs_titles[area_idx])
        
        for ax in axs:
            ax.legend(handles=handles, labels=['PreLesion', 'PostLesion'], loc='upper right', fontsize=10)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            if n_components == 3:
                ax.set_zlabel('PC3')
    elif fixed_field == 'conditions':
        axs_titles = ['PreLesion', 'PostLesion']
        handles = [None, None]

        colors = ['#0000FF', '#ff4800']
        colors_markers = [['#030426', '#0091ff'], ['#7a1b01', '#ff6f00']]

        for area_idx in np.arange(np.shape(NPD)[1]):
            color = colors[area_idx]
            for condition_idx in np.arange(np.shape(NPD)[0]):
                ax = axs[condition_idx]
                for trial_idx in np.arange(np.shape(NPD)[3]):
                    if n_components == 3:
                        handles[area_idx], = ax.plot(
                            NPD[condition_idx, area_idx, 0, trial_idx, trial_range],
                            NPD[condition_idx, area_idx, 1, trial_idx, trial_range],
                            NPD[condition_idx, area_idx, 2, trial_idx, trial_range],
                            color=color,
                            linewidth=0.5
                        )
                        ax.scatter(
                            NPD[condition_idx, area_idx, 0, trial_idx, trial_range[0]], 
                            NPD[condition_idx, area_idx, 1, trial_idx, trial_range[0]], 
                            NPD[condition_idx, area_idx, 2, trial_idx, trial_range[0]],
                            color=colors_markers[area_idx][0],
                            s=8,
                            marker='*',

                        )
                        ax.scatter(
                            NPD[condition_idx, area_idx, 0, trial_idx, trial_range[-1]], 
                            NPD[condition_idx, area_idx, 1, trial_idx, trial_range[-1]], 
                            NPD[condition_idx, area_idx, 2, trial_idx, trial_range[-1]],
                            color=colors_markers[area_idx][1],
                            s=8,
                            marker='<'
                        )
                    else:    
                        handles[area_idx], = ax.plot(
                            NPD[condition_idx, area_idx, 0, trial_idx, trial_range],
                            NPD[condition_idx, area_idx, 1, trial_idx, trial_range],
                            color=color,
                            linewidth=0.5
                        )
                        ax.scatter(
                            NPD[condition_idx, area_idx, 0, trial_idx, trial_range[0]], 
                            NPD[condition_idx, area_idx, 1, trial_idx, trial_range[0]],
                            color=colors_markers[area_idx][0],
                            s=8,
                            marker='*',

                        )
                        ax.scatter(
                            NPD[condition_idx, area_idx, 0, trial_idx, trial_range[-1]], 
                            NPD[condition_idx, area_idx, 1, trial_idx, trial_range[-1]],
                            color=colors_markers[area_idx][1],
                            s=8,
                            marker='<'
                        )
                
                ax.set_title(axs_titles[condition_idx])

        for ax in axs:
            ax.legend(handles=handles, labels=['RFA', 'S1'], loc='upper right', fontsize=10)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            if n_components == 3:
                ax.set_zlabel('PC3')

        ns.visualization.pyplot.tight_layout()
    else:
        raise ValueError("'" + fixed_field + "' cannot be fixed.")

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.dist = 10

    ns.visualization.pyplot.suptitle(title)
    ns.visualization.pyplot.tight_layout()

    return
