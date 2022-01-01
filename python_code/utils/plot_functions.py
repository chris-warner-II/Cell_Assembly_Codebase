import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp

from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpp

import numpy as np
from scipy import signal as sig

import sys
import os
import utils.retina_computation as rc
import time                         # to time operations for code analysis

from sklearn.metrics.pairwise import cosine_similarity 



# # Example of making your own norm.  Also see matplotlib.colors.
# # From Joe Kington: This one gives two different linear ramps:
# class MidpointNormalize(plt.colors.Normalize):
#     def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#         self.midpoint = midpoint
#         plt.colors.Normalize.__init__(self, vmin, vmax, clip)

#     def __call__(self, value, clip=None):
#         # I'm ignoring masked values and all kinds of edge cases to make a
#         # simple example...
#         x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#         return numpy.ma.masked_array(numpy.interp(value, x, y))




def hist_dataGen_andSampling_statistics(Ycell_hist1, Zassem_hist1, nY1, nZ1, CA_coactivity1, Cell_coactivity1, numSamp1, 
                                        Ycell_hist2, Zassem_hist2, nY2, nZ2, CA_coactivity2, Cell_coactivity2, numSamp2, 
                                        CA_ovl, ria, riap, N, M, Kmax, r, plt_save_dir, fname_save_tag, nameIdentifiers, figSaveFileType):


    # compute histograms from nY, nZ, nY_Samp and nZ_Samp
    binns = np.arange( np.concatenate([nY1, nY2]).min(), np.concatenate([nY1, nY2]).max()+2 )
    nYh1 = np.histogram(nY1,bins=binns)
    nYh2 = np.histogram(nY2,bins=binns)
    #
    binns = np.arange( np.concatenate([nZ1, nZ2]).min(), np.concatenate([nZ1, nZ2]).max()+2 )
    nZh1 = np.histogram(nZ1,bins=binns)
    nZh2 = np.histogram(nZ2,bins=binns)    



    #plt.figure( figsize=(20,10) ) # size units in inches
    plt.rc('font', weight='bold', size=12)
    f, ax = plt.subplots(2,4)
    f.set_size_inches( (20,10) )
    plt.suptitle( str('Data Biases? Compare ' + nameIdentifiers[0] + ' vs. ' + nameIdentifiers[1] + ' ' + nameIdentifiers[2] + ' data') )
    #
    ax[0][0].plot(100*Ycell_hist1[:-1]/numSamp1, 'rx--', linewidth=3, markersize=8, label=str('in ' + str(numSamp1) + ' ' + nameIdentifiers[0] + ' data'))
    ax[0][0].plot(100*Ycell_hist2[:-1]/numSamp2, 'bo-', linewidth=3, markersize=8, label=str('in ' + str(numSamp2) + ' ' + nameIdentifiers[1] + ' data') )
    ax[0][0].set_title('Cells $Y_i$')
    ax[0][0].set_ylabel('% Data')
    ax[0][0].set_xlabel('cell id')
    ax[0][0].set_xticks(np.floor(np.linspace(0,N,5)).astype(int))
    ax[0][0].grid()
    ax[0][0].legend(fontsize=10)
    #
    ax[0][1].plot(nYh1[1][:-1],       100*nYh1[0]/numSamp1, 'rx--', linewidth=3, markersize=8)
    ax[0][1].plot(nYh2[1][:-1] + 0.1, 100*nYh2[0]/numSamp2, 'bo-', linewidth=3, markersize=8)
    ax[0][1].set_title('Num Active Cells |Y|')
    ax[0][1].set_ylabel('% Data')
    ax[0][1].grid()
    #
    ax[0][2].axis('off')
    #
    ax[1][0].plot(100*Zassem_hist1[:-1]/numSamp1, 'rx--', linewidth=3, markersize=8)  
    ax[1][0].plot(100*Zassem_hist2[:-1]/numSamp2, 'bo-', linewidth=3, markersize=8) 
    ax[1][0].set_xlabel('Cell Assemblies $Z_a$')
    #ax[1][0].set_xlabel('CA id')
    ax[1][0].set_xticks(np.floor(np.linspace(0,M,3)).astype(int))
    ax[1][0].grid()
    #
    ax[1][1].plot(nZh1[1][:-1],       100*nZh1[0]/numSamp1, 'rx--', linewidth=3, markersize=8)
    ax[1][1].plot(nZh2[1][:-1] + 0.1, 100*nZh2[0]/numSamp2, 'bo-', linewidth=3, markersize=8)
    ax[1][1].set_xlabel('Num Active Assemblies |Z|')
    ax[1][1].grid()
    #
    Pia = rc.sig(ria)
    Piap = rc.sig(riap)
    bins = np.linspace(0, np.ceil( np.max( [ (1-Pia).sum(axis=0).max(), (1-Piap).sum(axis=0).max() ] ) ), 10)
    CperA_gt = np.histogram( (1-Pia).sum(axis=0), bins=bins )
    CperA_md = np.histogram( (1-Piap).sum(axis=0), bins=bins )
    ax[1][2].plot( CperA_gt[1][1:], CperA_gt[0], 'bo--', linewidth=3, markersize=8, label='Pia' )
    ax[1][2].plot( CperA_md[1][1:], CperA_md[0], 'rx--', linewidth=3, markersize=8, label='Piap' )
    ax[1][2].set_xlabel(' CA membership strength - $\sum_i P_{ia}$ ')
    ax[1][2].set_ylabel( str('Counts / '+str(M)+' CAs') )
    ax[1][2].legend(fontsize=10)
    #ax[1][2].set_title('Histogram Cells per CA ')




    ax5 = plt.subplot2grid((3,4),(0,3)) 
    im1=ax5.imshow(100*CA_coactivity1/numSamp1)
    ax5.set_xticks(np.floor(np.linspace(0,M-1,3)).astype(int))
    ax5.set_yticks(np.floor(np.linspace(0,M-1,3)).astype(int))
    cax1 = f.add_axes([0.90, 0.68, 0.02, 0.2])
    f.colorbar(im1, cax=cax1)
    cax1.set_title('% SWs')
    ax5.set_ylabel( str('CA Coact. in ' + nameIdentifiers[0] ) )


    ax6 = plt.subplot2grid((3,4),(1,3)) 
    im2=ax6.imshow(100*CA_coactivity2/numSamp2)
    ax6.set_xticks(np.floor(np.linspace(0,M-1,3)).astype(int))
    ax6.set_yticks(np.floor(np.linspace(0,M-1,3)).astype(int))
    cax2 = f.add_axes([0.90, 0.40, 0.02, 0.2])
    f.colorbar(im2, cax=cax2)
    cax2.set_title('% SWs')
    ax6.set_ylabel( str('CA Coact. in ' + nameIdentifiers[1] ) )


    #
    ax7 = plt.subplot2grid((3,4),(2,3)) 
    im3=ax7.imshow(CA_ovl)
    ax7.set_xticks(np.floor(np.linspace(0,M-1,3)).astype(int))
    ax7.set_yticks(np.floor(np.linspace(0,M-1,3)).astype(int))
    cax3 = f.add_axes([0.90, 0.1, 0.02, 0.2])
    f.colorbar(im3, cax=cax3)
    cax3.set_title('Dot Prod')
    ax7.set_ylabel( str('CA Overlap' ) )


    plt.tight_layout()
    #
    if not os.path.exists( str(plt_save_dir + 'DataGenSampStats/') ):
        os.makedirs( str(plt_save_dir + 'DataGenSampStats/') )
    plt.savefig( str(plt_save_dir + 'DataGenSampStats/' + 'DataDists_' + nameIdentifiers[0] +  '_vs_' + nameIdentifiers[1] + '_' + nameIdentifiers[2] + fname_save_tag + '.' + figSaveFileType ) ) # fname_save_tag + '_'
    plt.close() 






def plot_CA_inference_performance(inferCA_Confusion, inferCell_Confusion, CA_ovl, CA_coactivity, 
            zInferSampled, zInferSampledT, Zassem_hist, yInferSampled, Kinf, KinfDiff, N, M, M_mod, 
            numSWs, params_init, params_init_param, r, plt_save_dir, fname_save_tag, pltSpecifierTag, figSaveFileType ):
            # translate_Tru2Lrn,translate_Tru2Tru, translate_Lrn2Tru,translate_Lrn2Lrn, 

    numSamps = Kinf.size
    Binns =  np.arange( np.array([Kinf-KinfDiff, Kinf]).min().astype(int), np.array([Kinf-KinfDiff, Kinf]).max().astype(int)+1) 
    print('Binns = ', Binns)
    p = plt.hist(Kinf-KinfDiff, bins=Binns )
    plt.close()
    print('p = ',p)

    f = plt.figure( figsize=(20,10) ) # size units in inches
    plt.rc('font', weight='bold', size=16)

    f.suptitle( str('Cell Assembly (CA) Inference Performance Statistics w/ ' + params_init + ' parameters ' + params_init_param) )
    

    #
    ax1 = plt.subplot2grid((2,3),(0,0)) 
    im1=ax1.imshow(inferCA_Confusion) # [np.hstack([ translate_Lrn2Lrn,M_mod  ] ) ]
    ax1.set_title('CA Inference Confusion Matrix')
    M_lbl = list(np.floor(np.linspace(0,M_mod-1,3)).astype(int))
    ax1.set_xticks( np.hstack( [np.floor(np.linspace(0,M_mod-1,3)), M_mod] ).astype(int) ) 
    M_lbl.append('Add')
    ax1.set_xticklabels( M_lbl )
    ax1.set_yticks( np.hstack( [np.floor(np.linspace(0,M_mod-1,3)), M_mod] ).astype(int) ) 
    M_lbl.remove('Add') 
    M_lbl.append('Drp')
    ax1.set_yticklabels( M_lbl)  
    ax1.set_xlabel(str('missed CA'))
    ax1.set_ylabel(str('replaced w/ CA'))
    cax1 = f.add_axes([0.33, 0.55, 0.01, 0.3])
    f.colorbar(im1, cax=cax1)       
    

    #
    ax2 = plt.subplot2grid((2,3),(0,1)) 
    im2 = ax2.imshow(CA_ovl,vmin=0,vmax=1) #,cmap='bone_r'
    ax2.set_title('True CA pair dot product')
    ax2.set_xticks(np.floor(np.linspace(0,M-1,3)).astype(int))
    ax2.set_yticks(np.floor(np.linspace(0,M-1,3)).astype(int))
    cax2 = f.add_axes([0.61, 0.55, 0.01, 0.3])
    f.colorbar(im2, cax=cax2, ticks=[0, 0.5, 1])
    

    #
    # WAIT, WHAT IS THIS PLOT SHOWING??
    ax3 = plt.subplot2grid((2,3),(1,0)) 
    im3=ax3.imshow(CA_coactivity) #,cmap='bone_r' # /Zassem_hist.mean()
    ax3.set_ylabel(str('CA Coactivity in ' + str(numSWs) + ' SWs')) #w/ mean activation ~ ' + '{:04.3f}'.format( Zassem_hist.mean()/numSWs ) +'% of SWs') )
    ax3.set_xticks(np.floor(np.linspace(0,M-1,3)).astype(int))
    ax3.set_yticks(np.floor(np.linspace(0,M-1,3)).astype(int))  
    cax3 = f.add_axes([0.33, 0.13, 0.01, 0.3])
    f.colorbar(im3, cax=cax3)



    #
    ax4 = plt.subplot2grid((2,4),(0,3))
    ax4.hist(Kinf, bins=Binns, align='left', color='red', label = 'Infrd z', alpha=0.3)
    ax4.hist(Kinf-KinfDiff, bins=Binns, align='left', color='blue', label='Data Smp', alpha=0.3  )
    ax4.set_title('|z|')
    #ax4.set_xlabel('# of 1''s in z-vector')
    ax4.set_ylabel( str('# Counts / ' + str(numSamps) + ' Inf samples') )
    ax4.set_xticks(Binns)
    ax4.legend(bbox_to_anchor=(0.99, 0.99), loc=1, borderaxespad=0.)
    ax4.grid()


    #
    ax5 = plt.subplot2grid((2,3),(1,1))
    ax5.scatter(range(M_mod),Zassem_hist[:-1]/numSWs,s=200,c='black',label=str('in gen data'),alpha=0.6)
    ax5.scatter(range(M_mod),zInferSampled[:-1]/numSamps,s=200,c='red',label=str('Infrd z'),alpha=0.6) # [translate_Lrn2Lrn]
    ax5.scatter(range(M),   zInferSampledT[:-1]/numSamps,s=200,c='blue',label=str('Data Smp'),alpha=0.6)
    #ax5.set_xticks(np.floor(np.linspace(0,np.max(M,M_mod)-1,3)).astype(int))
    ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax5.set_yticks( np.linspace(0, np.max( ( (zInferSampled[:-1]/numSamps).max(), (zInferSampledT[:-1]/numSamps).max(), (Zassem_hist/numSWs).max() ) )*1.4, 3) ) 
    ax5.set_title('% CA Activity')
    ax5.set_xlabel('CA id')
    ax5.grid()
    ax5.legend() #bbox_to_anchor=(0.99, 0.01), loc=4, borderaxespad=0.)

    #
    ax6 = plt.subplot2grid((2,3),(1,2))
    im6=ax6.imshow(inferCell_Confusion)
    ax6.set_title('Cell Coactivity in $p(y_i|z)$')
    # ax6.set_xticks(np.floor(np.linspace(0,N,3)).astype(int))
    # ax6.set_yticks(np.floor(np.linspace(0,N,3)).astype(int))  
    N_lbl = list(np.floor(np.linspace(0,N,3)).astype(int))
    ax6.set_xticks(  np.hstack( [np.floor(np.linspace(0,N,3))] ).astype(int)  )
    #N_lbl.append('Add') 
    ax6.set_xticklabels( N_lbl )
    ax6.set_yticks(  np.hstack( [np.floor(np.linspace(0,N,3)), N] ).astype(int)  )  
    #N_lbl.remove('Add')
    #N_lbl.append('Drp')
    ax6.set_yticklabels( N_lbl ) 


    ax6.set_xlabel(str('y id'))
    #ax6.set_ylabel(str('replaced w/ y'))
    cax6 = f.add_axes([0.90, 0.13, 0.01, 0.3])
    f.colorbar(im6, cax=cax6)   



    #
    plt.tight_layout()
    if not os.path.exists( str(plt_save_dir+'InferencePerformance/')  ):
        os.makedirs( str(plt_save_dir+'InferencePerformance/') )
    plt.savefig( str(plt_save_dir + 'InferencePerformance/' + 'Confus_Coactiv_' + pltSpecifierTag + '_' + fname_save_tag + '.' + figSaveFileType ) )
    plt.close()    







def hist_SWsampling4learning_stats(smp, numSWs, ria, ri, C, Cmin, Cmax, r, plt_save_dir, fname_save_tag, fnameTag, figSaveFileType):

    numSamps = len(smp)

    N = ria.shape[0]
    M = ria.shape[1]

    Pi = rc.sig(ri)
    Pia = rc.sig(ria)

    plt.rc('font', weight='bold', size=18)
    f, ax = plt.subplots( 3,1, figsize=(20,10) )
    plt.subplots_adjust(hspace=0.4)
    #
    f.suptitle( str('Uniform Sampling? Sanity Checks :: ' + str(numSamps) +  ' ' + fnameTag +' Data Samples from ' + str(numSWs) + ' Spike words ') )
    #
    ax[0].grid()
    ax[0].hist(smp, bins=range(numSWs), color='black')
    ax[0].set_title('Uniform Sampling from Generated Spike words')
    ax[0].set_xlabel('spike word id')
    ax[0].set_ylabel('# samples')
    #
    ax[1].grid()
    ax[1].bar(range(M), (1-Pia).sum(axis=0), color='black')
    ax[1].plot([0,M],[C,C],'r--')
    ax[1].plot([0,M],[Cmin,Cmin],'r--')
    ax[1].plot([0,M],[Cmax,Cmax],'r--')
    ax[1].text(M-1,C,'C',color='red')
    ax[1].set_xticks( range(M-1) )
    ax[1].set_xlabel('cell assembly id')
    ax[1].set_ylabel('# cells / assembly')
    #
    ax[2].grid()
    ax[2].bar(range(N), (1-Pia).sum(axis=1), color='black')
    ax[2].plot(range(N),(1-Pi),'r--')
    ax[2].text(N-1,1-Pi[-1],'1-Pi',color='red')
    ax[2].set_xticks( range(N) )
    ax[2].set_xlabel('cell id')
    ax[2].set_ylabel('# assemblies / cell')
    #
    plt.tight_layout()
    #
    if not os.path.exists(plt_save_dir):
        os.makedirs(plt_save_dir)
    plt.savefig( str(plt_save_dir + 'Sampling4Learning_' + fname_save_tag + '_' + str(numSamps) + '_' + fnameTag +'_rand' + str(r) + '.' + figSaveFileType) )

    plt.close()




def plot_params_MSE_during_learning(q_MSE, ri_MSE, ria_MSE, samps2snapshot, numSamps, N, M, learning_rate, lRateScale_PiQ, params_init, params_init_param, r, plt_save_dir, fname_save_tag, figSaveFileType):

    # plt.figure( figsize=(20,10) ) # size units in inches
    # plt.rc('font', weight='bold', size=20)

    # print(q_MSE.size)
    # print(samps2snapshot)

    samps2snapshot = samps2snapshot[samps2snapshot<q_MSE.size]

    x_ria = (np.linspace(1,numSamps,ria_MSE.shape[0])-1).astype(int)

    plt.rc('font', weight='bold', size=30)
    f,ax = plt.subplots(1,1, figsize=(20,10))
    #
    ax.plot(x_ria, np.log(ria_MSE[:,0]), 'k-', linewidth=2, label='$\mu$')
    ax.plot(x_ria, np.log(ria_MSE[:,0] + ria_MSE[:,1]), 'k--', linewidth=2, label='$\mu \pm \sigma$')
    ax.plot(x_ria, np.log(ria_MSE[:,2]), 'k.-.', linewidth=2, label='$max$')
    #
    ax.plot(x_ria, np.log(ria_MSE[:,0]), 'r-', linewidth=2)
    ax.plot(x_ria, np.log(ria_MSE[:,0] + ria_MSE[:,1]), 'r--', linewidth=2)
    #ax.plot(x_ria, np.log(ria_MSE[:,0] - ria_MSE[:,1]), 'r--', linewidth=2)
    ax.plot(x_ria, np.log(ria_MSE[:,2]), 'r.-.', linewidth=2)
    #
    ax.plot(samps2snapshot, np.log(ri_MSE[samps2snapshot,0]), 'b-', linewidth=2)
    ax.plot(samps2snapshot, np.log(ri_MSE[samps2snapshot,0] + ri_MSE[samps2snapshot,1]), 'b--', linewidth=2)
    #ax.plot(samps2snapshot, np.log(ri_MSE[samps2snapshot,0] - ri_MSE[samps2snapshot,1]), 'b--', linewidth=2)
    ax.plot(samps2snapshot, np.log(ri_MSE[samps2snapshot,2]), 'b.-.', linewidth=2)
    #
    ax.plot(samps2snapshot,np.log(q_MSE[samps2snapshot]),'g.-.', linewidth=2)
    #
    xTxt = int(numSamps*2/3)  # can include true model params here in text.
    ax.text( 0.6*numSamps, np.log(ria_MSE).max(), '$r_{ia}$', color='r' ) # C, Cmin, Cmax, sigPia
    ax.text( 0.7*numSamps, np.log(ria_MSE).max(), '$r_{i}$', color='b' )    # K, Kmin, Kmax, muPi, SigPi
    ax.text( 0.8*numSamps, np.log(ria_MSE).max(), '$q$', color='g' )                            # '=K/M ~ 0.0009' 
    #
    ax.legend(fontsize='x-small')
    ax.set_title(str('MSE Params w/ Init ' + params_init + params_init_param.replace('pt','.').replace('_',' ') + ' (N=' + str(N) + ', M=' + str(M) + ')'))
    ax.set_xlabel(str('Learning Step : LR=' + str(learning_rate)  + ' LRpi' + str(lRateScale_PiQ) ))
    ax.set_ylabel('$log MSE $')
    ax.grid()
    #
    plt.tight_layout()
    #
    if not os.path.exists( str(plt_save_dir + 'ParamsMSE/') ):
        os.makedirs( str(plt_save_dir + 'ParamsMSE/') )
    plt.savefig( str(plt_save_dir  + 'ParamsMSE/' + 'ParamsMSE_' + fname_save_tag + '.' + figSaveFileType ) )
    plt.close()




def plot_params_Err_during_learning(Q_SE, Pi_SE, Pi_AE, Pia_AE, PiaOn_SE, PiaOn_AE, PiaOff_SE, PiaOff_AE, samps2snapshot, \
    numSamps, N, M, learning_rate, lRateScale_PiQ, params_init, params_init_param, r, plt_save_dir, fname_save_tag):

    # NEED TO ADD: NUM_DPSIG_SNAPS !!! It is computed with all:
    #   Pia_AE, PiaOn_SE, PiaOn_AE, PiaOff_SE, PiaOff_AE, num_dpSig_snaps 


    samps2snapshot = samps2snapshot[samps2snapshot<Q_SE.size]

    x_Pia = (np.linspace(1,numSamps,Pia_AE.shape[0])-1).astype(int)

    plt.rc('font', weight='bold', size=20)
    f,ax = plt.subplots(2,1, figsize=(20,10))
    #
    ax[0].plot(x_Pia, np.log(Pia_AE[:,0]), 'k-', linewidth=2, label='$\mu$') # THIS IS FOR LEGEND.
    ax[0].plot(x_Pia, np.log(Pia_AE[:,0] + Pia_AE[:,1]), 'k--', linewidth=2, label='$\mu + \sigma$')
    #ax[0].plot(x_Pia, np.log(Pia_AE[:,2]), 'k.-.', linewidth=2, label='$max$')
    #
    ax[0].plot(x_Pia, np.log(Pia_AE[:,0]), 'r-', linewidth=2) #, label='Pia All'
    ax[0].plot(x_Pia, np.log(Pia_AE[:,0] + Pia_AE[:,1]), 'r--', linewidth=2)
    #ax[0].plot(x_Pia, np.log(Pia_AE[:,2]), 'r.-.', linewidth=2)
    #
    ax[0].plot(x_Pia, np.log(PiaOn_AE[:,0]), 'm-', linewidth=2) #, label='Pia in CA'
    ax[0].plot(x_Pia, np.log(PiaOn_AE[:,0] + PiaOn_AE[:,1]), 'm--', linewidth=2)
    #ax[0].plot(x_Pia, np.log(PiaOn_AE[:,2]), 'm.-.', linewidth=2)
    #
    ax[0].plot(x_Pia, np.log(PiaOff_AE[:,0]), 'c-', linewidth=2) #, label='Pia out'
    ax[0].plot(x_Pia, np.log(PiaOff_AE[:,0] + PiaOn_AE[:,1]), 'c--', linewidth=2)
    #ax[0].plot(x_Pia, np.log(PiaOff_AE[:,2]), 'c.-.', linewidth=2)
    #
    ax[0].plot(samps2snapshot, np.log(Pi_AE[samps2snapshot,0]), 'b-', linewidth=2) #, label='Pi'
    ax[0].plot(samps2snapshot, np.log(Pi_AE[samps2snapshot,0] + Pi_AE[samps2snapshot,1]), 'b--', linewidth=2)
    #ax[0].plot(samps2snapshot, np.log(Pi_AE[samps2snapshot,2]), 'b.-.', linewidth=2)
    #
    ax[0].plot(samps2snapshot,np.log(np.abs(Q_SE[samps2snapshot])),'g.-.', linewidth=2) #, label='Q'
    #
    # xTxt = int(numSamps*2/3)  # can include true model params here in text.
    # ax[0].text( 0.5*numSamps,  0.8*np.log(np.abs(Q_SE)).min(), '$P_{ia}$ in CA', color='m' ) # C, Cmin, Cmax, sigPia
    # ax[0].text( 0.65*numSamps, 0.8*np.log(np.abs(Q_SE)).min(), '$P_{ia}$ not CA', color='c' ) # C, Cmin, Cmax, sigPia
    # ax[0].text( 0.5*numSamps, 0.7*np.log(np.abs(Q_SE)).min(), '$P_{ia}$', color='r' ) # C, Cmin, Cmax, sigPia
    # ax[0].text( 0.6*numSamps, 0.7*np.log(np.abs(Q_SE)).min(), '$P_{i}$', color='b' )    # K, Kmin, Kmax, muPi, SigPi
    # ax[0].text( 0.7*numSamps, 0.7*np.log(np.abs(Q_SE)).min(), '$Q$', color='g' )                            # '=K/M ~ 0.0009' 
    #
    ax[0].legend(fontsize='x-small')
    #ax[0].set_xlabel(str('Learning Step : LR=' + str(learning_rate)  + ' LR Pi and Q' + str(lRateScale_PiQ) ))
    ax[0].set_ylabel('log Err of Abs Val')
    ax[0].grid()
    #
    # #
    # # # # # # # #
    # #
    #

    #Q_SE, Pi_SE, PiaOn_SE, PiaOff_SE, # --- PiaOn_AE, PiaOff_AE
    # mean, std, max, min, numElements (for sem instead of std)

    ax[1].errorbar(1.0*x_Pia, Pia_AE[:,0], Pia_AE[:,1]/np.sqrt(Pia_AE[:,3]), color='r', capsize=5, capthick=3)   # mean & std errorbar.
    ax[1].plot(1.0*x_Pia, Pia_AE[:,0], 'r--', linewidth=3, label='|1-Pia|')      # mean
    #
    # For Pia entries where Cells are Members of CA.
    ax[1].errorbar(1.1*x_Pia, PiaOn_SE[:,0], PiaOn_SE[:,1]/np.sqrt(PiaOn_SE[:,4]), color='m', capsize=5, capthick=3)   # mean & std errorbar.
    ax[1].plot(1.1*x_Pia, PiaOn_SE[:,0], 'm--', linewidth=3, label='CA member')      # mean
    # ax[1].plot(1.1*x_Pia, PiaOn_SE[:,2], 'm.-.', linewidth=1)                       # max
    # ax[1].plot(1.1*x_Pia, PiaOn_SE[:,3], 'm.-.', linewidth=1)                       # min
    #
    # For Pia entries where Cells are not Members of CA. HERE!!
    ax[1].errorbar(1.2*x_Pia, PiaOff_SE[:,0], PiaOff_SE[:,1]/np.sqrt(PiaOff_SE[:,4]), color='c', capsize=5, capthick=3)   # mean & std errorbar.
    ax[1].plot(1.2*x_Pia, PiaOff_SE[:,0], 'c--', linewidth=3, label='not in CA')       # mean
    # ax[1].plot(1.2*x_Pia, PiaOff_SE[:,2], 'c.-.', linewidth=1)                        # max
    # ax[1].plot(1.2*x_Pia, PiaOff_SE[:,3], 'c.-.', linewidth=1)                        # min
    #
    ax[1].errorbar(samps2snapshot, Pi_SE[samps2snapshot,0], Pi_SE[samps2snapshot,1]/np.sqrt(Pi_SE[samps2snapshot,4]), color='b', capsize=5, capthick=3)   # mean & std errorbar.
    ax[1].plot(samps2snapshot, Pi_SE[samps2snapshot,0], 'b--', linewidth=3, label='1-Pi')                                   # mean
    # ax[1].plot(samps2snapshot, Pi_SE[samps2snapshot,2], 'b.-.', linewidth=1)                                 # max
    # ax[1].plot(samps2snapshot, Pi_SE[samps2snapshot,3], 'b.-.', linewidth=1)                                 # min
    #

    ax[1].plot(samps2snapshot,Q_SE[samps2snapshot],'g--', linewidth=3, label='Q')
    # #
    ax[1].legend(fontsize='x-small')
    ax[1].set_xlabel(str('Learning Step : LR=' + str(learning_rate)  + ' LR Pi/Q' + str(lRateScale_PiQ) ))
    ax[1].set_ylabel('Signed Error (GT-Mod)')
    ax[1].grid()
    #
    plt.tight_layout()
    plt.suptitle(str('Error in Params w/ Init ' + params_init + params_init_param.replace('pt','.').replace('_',' ') + ' (N=' + str(N) + ', M=' + str(M) + ')'))
    
    if not os.path.exists( str(plt_save_dir + 'ParamsErr/') ):
        os.makedirs( str(plt_save_dir + 'ParamsErr/') )
    plt.savefig( str(plt_save_dir  + 'ParamsErr/' + 'ParamsErr_' + fname_save_tag + '.' + figSaveFileType ) )
    plt.close()




def plot_params_derivs_during_learning(q_deriv, ri_deriv, ria_deriv, numSamps, N, M, learning_rate, lRateScale, ds_fctr, params_init, params_init_param, r, plt_save_dir, fname_save_tag, figSaveFileType):
    #
    # ria_deriv & ri_deriv contain: [   [0]. np.sign(dria.mean())*np.abs(dria).mean(),     
    #                                   [1]. np.abs(dria).std(), 
    #                                   [2]. np.abs(dria).max(),
    #                                   [3]. dria.mean(),    
    #                                   [4]. dria.std(),     
    #                                   [5]. dria.max(),         
    #                                   [6]. dria.min()  ]

    numSamps = np.min([numSamps,q_deriv.size])

    ds_fctr = int(ds_fctr) # smooth jumpy derivatives by averaging over a number of samples ( convolving below.)
    x = np.linspace(0,numSamps-1, int(numSamps/ds_fctr) ).round().astype(int)[:-1]
    #
    f=plt.figure( figsize=(20,10) ) # size units in inches
    plt.rc('font', weight='bold', size=12)
    ax = f.subplots(2,3)
    f.suptitle(str('Using parameter derivatives to determine convergence w/ LR =' + str(learning_rate) + ' & LRsc =' + str(lRateScale) ))
    #
    y = np.convolve(np.abs(ria_deriv[:,0]), np.ones((ds_fctr,))/ds_fctr, mode='valid')
    e = np.convolve(ria_deriv[:,1], np.ones((ds_fctr,))/ds_fctr, mode='valid')
    ax[0][0].plot(x, y[x], color='red', label='$\mu \pm \sigma (|.|)$')
    ax[0][0].fill_between(x, y[x]-e[x], y[x]+e[x], alpha=0.2, edgecolor='red', facecolor='red')
    y = np.convolve(ria_deriv[:,3], np.ones((ds_fctr,))/ds_fctr, mode='valid')
    e = np.convolve(ria_deriv[:,4], np.ones((ds_fctr,))/ds_fctr, mode='valid')           
    ax[0][0].plot(x, y[x], color='blue', label='$\mu \pm \sigma$')
    ax[0][0].fill_between(x, y[x]-e[x], y[x]+e[x], alpha=0.2, edgecolor='blue', facecolor='blue')
    ax[0][0].plot(x,np.zeros_like(x),'k--')
    ax[0][0].grid()
    ax[0][0].set_ylabel('dria')
    #ax[0][0].legend(fontsize=6)
    #
    y = np.convolve(ria_deriv[:,5], np.ones((ds_fctr,))/ds_fctr, mode='valid')
    ax[0][1].scatter(x, y[x], s=20, label='$max$', alpha=0.5)
    y = np.convolve(ria_deriv[:,6], np.ones((ds_fctr,))/ds_fctr, mode='valid')
    ax[0][1].scatter(x, y[x], s=20, label='$min$', alpha=0.5)
    y = np.convolve(ria_deriv[:,2], np.ones((ds_fctr,))/ds_fctr, mode='valid')
    ax[0][1].scatter(x, y[x], s=20, label='$max(|.|)$', alpha=0.5)
    ax[0][1].plot(x,np.zeros_like(x),'k--')
    ax[0][1].grid()
    ax[0][1].set_ylabel('dria')
    ax[0][1].set_xlabel('EM iteration')
    #ax[0][1].legend(fontsize=6)
    #
    y = np.convolve(ria_deriv[:,0], np.ones((ds_fctr,))/ds_fctr, mode='valid')
    ax[0][2].scatter(x, y[x], color='green', s=20, label='$sign*\mu(|.|)$', alpha=0.5)
    ax[0][2].plot(x,np.zeros_like(x),'k--')
    ax[0][2].grid()
    ax[0][2].set_ylabel('dria')
    #ax[0][2].legend(fontsize=6)
    #
    # #
    #       
    y = np.convolve(np.abs(ri_deriv[:,0]), np.ones((ds_fctr,))/ds_fctr, mode='valid')
    e = np.convolve(ri_deriv[:,1], np.ones((ds_fctr,))/ds_fctr, mode='valid')
    ax[1][0].plot(x, y[x], color='red', label='$\mu \pm \sigma (|.|)$')
    ax[1][0].fill_between(x, y[x]-e[x], y[x]+e[x], alpha=0.2, edgecolor='red', facecolor='red')
    y = np.convolve(ri_deriv[:,3], np.ones((ds_fctr,))/ds_fctr, mode='valid')
    e = np.convolve(ri_deriv[:,4], np.ones((ds_fctr,))/ds_fctr, mode='valid')            
    ax[1][0].plot(x, y[x], color='blue', label='$\mu \pm \sigma$')
    ax[1][0].fill_between(x, y[x]-e[x], y[x]+e[x], alpha=0.2, edgecolor='blue', facecolor='blue')
    ax[1][0].plot(x,np.zeros_like(x),'k--')
    ax[1][0].grid()
    ax[1][0].set_ylabel('dri')
    ax[1][0].legend()#fontsize=6)
    #
    y = np.convolve(ri_deriv[:,5], np.ones((ds_fctr,))/ds_fctr, mode='valid')
    ax[1][1].scatter(x, y[x], s=20, label='$max$', alpha=0.5)
    y = np.convolve(ri_deriv[:,6], np.ones((ds_fctr,))/ds_fctr, mode='valid')
    ax[1][1].scatter(x, y[x], s=20, label='$min$', alpha=0.5)
    y = np.convolve(ri_deriv[:,2], np.ones((ds_fctr,))/ds_fctr, mode='valid')
    ax[1][1].scatter(x, y[x], s=20, label='$max(|.|)$', alpha=0.5)
    ax[1][1].plot(x,np.zeros_like(x),'k--')
    ax[1][1].grid()
    ax[1][1].set_ylabel('dri')
    ax[1][1].legend()#fontsize=6)
    #
    y = np.convolve(ri_deriv[:,0], np.ones((ds_fctr,))/ds_fctr, mode='valid')
    ax[1][2].scatter(x, y[x], color='green', s=20, label='$sign*\mu(|.|)$', alpha=0.5)
    y = np.convolve(q_deriv, np.ones((ds_fctr,))/ds_fctr, mode='valid')
    ax[1][2].scatter(x, y[x], color='red', s=20, label='$dq$', alpha=0.5)
    ax[1][2].plot(x,np.zeros_like(x),'k--')
    ax[1][2].grid()
    ax[1][2].set_ylabel('dri & dq')
    ax[1][2].legend()#fontsize=6)
    #       
    plt.tight_layout()
    #
    if not os.path.exists( str(plt_save_dir+'ParamsDerivs/')):
        os.makedirs( str(plt_save_dir+'ParamsDerivs/') )
    plt.savefig( str( plt_save_dir + 'ParamsDerivs/' + 'ParamsDerivs_' + fname_save_tag + '.' + figSaveFileType ) )
    plt.close() 



# MAKE NEW SUBPLOT ROUTINES ??? NOT NOW. MAYBE LATER.
#def imshw(M,handle,xtick,ytick,xlabel,ylabel,title,fontsize,vmin,vmax,cmap,cbar=False):




def plot_params_init_n_learned(q, ri, ria, qp, rip, riap, q_init, ri_init, ria_init, zInferSampled,   
            numSamps, N, M, M_mod, params_init, params_init_param, r, plt_save_dir, fname_save_tag, figSaveFileType):
            # learning_rate, lRateScale_PiQ, 
            # translate_Tru2Lrn, translate_Tru2Tru, translate_Lrn2Tru, translate_Lrn2Lrn,
    

    # # # # SORT CELL ASSEMLIES BY ACTIVITY # # # # # # # 
    #
    # xx = np.argsort(zInferSampled)[::-1] - could use this to index into Pia's [:,xx] and sort 
    # them by the number of times they are inferred. But it hides interesting information about
    # model being overcomplete or undercomplete. Not doing now.


    #     
    # Converting model params into probabilities because Comparison is really weird.
    Q_init      = rc.sig(q_init)
    Pi_init     = rc.sig(ri_init)
    Pia_init    = rc.sig(ria_init)
    #
    Q           = rc.sig(q)
    Pi          = rc.sig(ri)
    Pia         = rc.sig(ria)
    #
    Qp          = rc.sig(qp)
    Pip         = rc.sig(rip)
    Piap        = rc.sig(riap)

    PiaGT      = np.where(Pia<.9999999)        # Pia ground truth (cell / assembly intersections)
    z_ntInf    = np.where(zInferSampled==0)[0]   # Cell assemblies (za's) that are never inferred
    Pi_wrong   = np.where(np.abs(Pip-Pi) > 0.1 )[0]                 # Cells (yi's) that have with wrong learned Pi values





    # print('Pia shape = ',Pia.shape)
    # print('Piap shape = ',Piap.shape)
    # print('Pia_init shape = ',Pia_init.shape)


    vmin = 0 #np.floor( np.array( [Pia.min(),Piap.min()] ).min() )
    vmid = 0.5
    vmax = 1 #np.ceil( np.array( [Pia.max(),Piap.max()] ).max() )

    if M<M_mod:
        CA_ticks = [0,M,M_mod]
        CA_tick_labels = ['0', str('M='+str(M)), str('Mmod='+str(M_mod))]
    elif M>M_mod:
        CA_ticks = [0,M_mod,M]
        CA_tick_labels = ['0', str('Mmod='+str(M_mod)), str('M='+str(M))]
    else:
        CA_ticks = [0,M]
        CA_tick_labels = ['0', str('M=Mmod='+str(M))]
   
    
    f=plt.figure( figsize=(20,10) ) # size units in inches
    plt.rc('font', weight='bold', size=14)
    ax = f.subplots(2,3)
    #
    # Plot Pia (true, init, learned)
    ax[0][0].scatter(PiaGT[1],PiaGT[0],s=30,color='none',edgecolors='black',marker='s',alpha=0.7)
    s=ax[0][0].imshow(Pia_init, vmin=vmin, vmax=vmax, cmap='gist_gray')
    ax[0][0].set_title('$P_{ia}$ Initialized')
    ax[0][0].set_xticks(CA_ticks)
    ax[0][0].set_yticks([0,N])
    ax[0][0].set_ylabel('cell ($y_i$)') 
    ax[0][0].grid()
    #
    ax[0][1].scatter(PiaGT[1],PiaGT[0],s=30,color='none',edgecolors='black',marker='s',alpha=0.7)
    s=ax[0][1].imshow(Pia, vmin=vmin, vmax=vmax, cmap='gist_gray')
    ax[0][1].set_title('$P_{ia}$ True')
    ax[0][1].set_xticks(CA_ticks)
    ax[0][1].set_yticks([0,N])   
    ax[0][1].grid()
    #
    ax[0][2].scatter(z_ntInf,N*np.ones_like(z_ntInf),s=40,color='red',edgecolors='black',alpha=0.3)
    ax[0][2].scatter(M_mod*np.ones_like(Pi_wrong),Pi_wrong,s=40,color='green',edgecolors='black',marker='d',alpha=0.3)
    ax[0][2].scatter(PiaGT[1],PiaGT[0],s=30,color='none',edgecolors='black',marker='s',alpha=0.7)
    s=ax[0][2].imshow(Piap, vmin=vmin, vmax=vmax, cmap='gist_gray')

    ax[0][2].set_title('$P_{ia}$ Learned')
    ax[0][2].set_xticks(CA_ticks)
    ax[0][2].set_yticks([0,N]) 
    ax[0][2].grid()   
    #
    cbar_ax = f.add_axes([0.91, 0.53, 0.02, 0.35])
    cb1=f.colorbar(s, cax=cbar_ax, ticks=[vmin, vmid, vmax])# )
    cb1.ax.set_title('$p(y_i=0|z_a=1)$',fontsize=14)
    cb1.ax.set_yticklabels([str( str(vmin)+' : active'), str(vmid), str( str(vmax)+' : silent') ],fontsize=10)
    #
    diff_init = Pia - Pia_init # Signed error (GT - Model)
    diff_learn = Pia - Piap
    # print(diff_init.min(), diff_init.max(), diff_learn.min(), diff_learn.max() )


    dmax = rc.sig( np.abs(np.vstack((diff_learn, diff_init)).max()))-0.5
    dmaxL = np.vstack((diff_learn, diff_init)).max()
    dmid = 0
    dmin = rc.sig(np.vstack((diff_learn, diff_init)).min())-0.5
    dminL = np.vstack((diff_learn, diff_init)).min()
    #
    ax[1][0].scatter(PiaGT[1],PiaGT[0],s=30,color='none',edgecolors='black',marker='s',alpha=0.7)
    d=ax[1][0].imshow(rc.sig(diff_init)-0.5, vmin=-dmax, vmax=dmax, cmap='bwr') # 
    ax[1][0].set_title('Err $P_{ia}$ at init')
    ax[1][0].set_xticks(CA_ticks)
    ax[1][0].set_xticklabels(CA_tick_labels)
    ax[1][0].set_yticks([0,N])
    ax[1][0].set_xlabel('assembly ($z_a$)')
    ax[1][0].set_ylabel('cell ($y_i$)')
    ax[1][0].grid()
    #
    ax[1][2].scatter(z_ntInf,N*np.ones_like(z_ntInf),s=40,color='red',edgecolors='black',alpha=0.1)
    ax[1][2].scatter(M_mod*np.ones_like(Pi_wrong),Pi_wrong,s=40,color='green',marker='d',edgecolors='black',alpha=0.1)
    ax[1][2].scatter(PiaGT[1],PiaGT[0],s=30,color='none',edgecolors='black',marker='s',alpha=0.7)
    d=ax[1][2].imshow(rc.sig(diff_learn)-0.5, vmin=-dmax, vmax=dmax, cmap='bwr') # 
    ax[1][2].set_title('Err $P_{ia}$ after learn')
    ax[1][2].set_xticks(CA_ticks)
    ax[1][2].set_xticklabels(CA_tick_labels)
    ax[1][2].set_yticks([0,N])
    ax[1][2].set_xlabel('assembly ($z_a$)')
    ax[1][2].grid()
    #ax[1][2].set_ylabel('cell ($y_i$)')
    cbar_ax2 = f.add_axes([0.91, 0.1, 0.02, 0.35])
    cb2 = f.colorbar(d, cax=cbar_ax2, ticks=[-dmax, -dmax/2, -dmax/4, dmid, dmax/4, dmax/2, dmax])
    cb2.ax.set_yticklabels([str( '-'+str(np.round(dmaxL,2))+' : mod<GT'), str('-'+str(np.round(dmaxL/2,2))), str('-'+str(np.round(dmaxL/4,2))),\
        str(np.round(dmid,2)), str(np.round(dmaxL/4,2)), str(np.round(dmaxL/2,2)), str( str(np.round(dmaxL,2))+' : mod>GT') ],fontsize=10)
    cb2.ax.set_title('sigmoid \n colors',fontsize=12)
    #
    # #
    #
    #ax0 = plt.subplot2grid( (10,3),(5,1), rowspan=1, colspan=1 )

    GS = gsp.GridSpec(10, 3, wspace=0.7)
    ax0 = plt.subplot(GS[5,1])
    # gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=GS[0])
    ax0.scatter( np.arange(len(zInferSampled)), zInferSampled, s=20, marker='d', color='cyan')
    ax0.set_ylabel("#inf'rd")
    ax0.autoscale(enable=True, axis='x', tight=True)
    #ax0.set_ylim(zInferSampled.min()-5,zInferSampled.max()+5)
    ax0.set_xticks(CA_ticks)
    ax0.set_xticklabels([]) #CA_tick_labels)
    ax0.grid()
    #
    # #
    #
    # Scatter plot initialized Pi,q and learned values to show their movement.
    vmin2 = 0 #np.floor( np.array( [ri.min(),rip.min(),q.min(),qp.min()] ).min() )
    vmid2 = 0.5
    vmax2 = 1 #np.ceil( np.array( [ri.max(),rip.max(),q.max(),qp.max()] ).max() )
    ax1 = plt.subplot(GS[6:,1])
    #plt.subplot2grid( (10,3),(6,1), rowspan=4, colspan=1 )
    ax1.plot([vmin2,vmax2],[vmin2,vmax2],'k--')
    ax1.scatter(Pi,Pi_init,s=20, color='black',label='init',alpha=0.3)
    ax1.scatter(Q,Q_init,s=20,color='black',alpha=0.3)
    ax1.scatter(Pi,Pip,s=80,color='blue',label='$r_i$ learned',alpha=0.5)
    #
    for ind in Pi_wrong:
        ax1.text(Pi[ind]-0.02,Pip[ind],str(ind),fontsize=14)
    #
    #Pi_noisy = np.where( Pi < 0.5 )
    #
    ax1.scatter(Q,Qp,s=80,color='green',label='$q$ learned',alpha=0.9)
    ax1.set_xticks([0,0.2,0.5,0.7,1])
    ax1.set_yticks([0,0.2,0.5,0.7,1])
    ax1.axis('equal')
    ax1.autoscale(enable=True, axis='both', tight=True)
    #ax1.set_xlim(0,1)
    #ax1.set_ylim(0,1)
    ax1.grid()
    ax1.set_xlabel('GT value')
    ax1.set_ylabel('model value')
    ax1.legend(fontsize='x-small')
    #
    plt.suptitle('Parameter fits @ Initialization and Post-learning', fontsize=30)
    plt.tight_layout()
    #
    if not os.path.exists( str(plt_save_dir+'EM_model_learned/') ):
        os.makedirs(str(plt_save_dir+'EM_model_learned/'))
    plt.savefig( str(plt_save_dir + 'EM_model_learned/' + 'EM_model_learned_' + fname_save_tag + '.' + figSaveFileType ) )
    plt.close()
    # plt.show()



def plot_compare_YinferVsTrueVsObs(numCellsCorrectInYobs, numCellsAddedInYobs, numCellsDroppedInYobs, numCellsTotalInYobs, \
        yCapture_Collect_Ordr1, yExtra_Collect_Ordr1, yMissed_Collect_Ordr1, yCapture_binVobs, yExtra_binVobs, yMissed_binVobs, \
        numSamps, Q, bernoulli_Pi, mu_Pi, sig_Pi, mu_Pia, sig_Pia, params_init, params_init_param, plt_save_dir, figSaveFileType):


    # # Variables to pass in
    # numCellsCorrectInYobs
    # numCellsAddedInYobs
    # numCellsDroppedInYobs
    # numCellsTotalInYobs
    # #
    # yCapture_Collect_Ordr1
    # yExtra_Collect_Ordr1
    # yMissed_Collect_Ordr1
    # #
    # yCapture_binVobs
    # yExtra_binVobs
    # yMissed_binVobs
    # #
    # numSamps
    # bernoulli_Pi
    # mu_Pi
    # sig_Pi
    # mu_Pia
    # sig_Pia
    # Q




    maxx = np.max( np.vstack([numCellsCorrectInYobs,numCellsAddedInYobs,numCellsDroppedInYobs]) )
    a = plt.hist(numCellsCorrectInYobs/numCellsTotalInYobs, bins=np.linspace(0,1,11))
    b = plt.hist(numCellsAddedInYobs/numCellsTotalInYobs, bins=np.linspace(0,1,11))
    c = plt.hist(numCellsDroppedInYobs/numCellsTotalInYobs, bins=np.linspace(0,1,11))
    #
    maxx = np.max( np.vstack([ (yCapture_Collect_Ordr1 - yCapture_binVobs), (yExtra_Collect_Ordr1 - yExtra_binVobs), (yMissed_Collect_Ordr1 - yMissed_binVobs)]) )
    minn = np.min( np.vstack([ (yCapture_Collect_Ordr1 - yCapture_binVobs), (yExtra_Collect_Ordr1 - yExtra_binVobs), (yMissed_Collect_Ordr1 - yMissed_binVobs)]) )
    d = plt.hist( yCapture_Collect_Ordr1 - yCapture_binVobs, bins=np.arange(minn,maxx+1))
    e = plt.hist( (yExtra_Collect_Ordr1 - yExtra_binVobs), bins=np.arange(minn,maxx+1))
    f = plt.hist( (yMissed_Collect_Ordr1 - yMissed_binVobs), bins=np.arange(minn,maxx+1))
    plt.close()
    #

    #plt.rc('font', weight='bold', size=16)
    g = plt.figure( figsize=(20,10) ) 
    g.suptitle(str('Spike Words ($y$) Observed with $P_i -> N(\mu='+ str(mu_Pi) +',\sigma=' + str(sig_Pi) + ')$; $P_{ia} -> N(\mu=' + str(mu_Pia) +',\sigma=' + str(sig_Pia) + ')$'))
    #
    ax1 = plt.subplot2grid((2,2),(0,0)) 
    ax1.plot(a[1][:-1],a[0], 'ro--', linewidth=3, markersize=16, label='% Correct ($\in P_{ia}$)') # markerfmt='ro', basefmt='k--',
    ax1.plot(b[1][:-1],b[0], 'go--', linewidth=3, markersize=16, label='% Added') # markerfmt='go', basefmt='k--',
    ax1.plot(c[1][:-1],c[0], 'bo--', linewidth=3, markersize=16, label='% Dropped') # markerfmt='bo', basefmt='k--',
    ax1.set_title( str('% cells $y_i$ observed in Cell Assembly') )
    ax1.set_xlabel( str('% cells') )
    ax1.set_ylabel( str('counts / ' + str(numSamps) + ' observations') )
    ax1.legend()
    ax1.grid()
    #
    ax2 = plt.subplot2grid((2,2),(0,1)) 
    ax2.plot(d[1][:-1],d[0], 'ro--', linewidth=3, markersize=16, label='% Correct ($\in P_{ia}$)') # markerfmt='ro', basefmt='k--',
    ax2.plot(e[1][:-1]-0.1,e[0], 'go--', linewidth=3, markersize=16, label='% Added') # markerfmt='go', basefmt='k--',
    ax2.plot(f[1][:-1]+0.1,f[0], 'bo--', linewidth=3, markersize=16, label='% Dropped') # markerfmt='bo', basefmt='k--',
    ax2.set_title( str('Comparing Inferred z vs "True" z - Represents observed y?') )
    ax2.set_xlabel( str('# observed cells modeled (Inferred - Ground Truth)') )
    ax2.set_ylabel( str('counts / ' + str(numSamps) + ' observations') )
    #ax2.legend()
    ax2.grid()
    #
    ax4 = plt.subplot2grid((2,3),(1,0)) 
    imCB2 = ax4.hist2d(numCellsAddedInYobs/numCellsTotalInYobs, numCellsTotalInYobs, bins = [ np.linspace(0,1,11), np.linspace(1,numCellsTotalInYobs.max(),numCellsTotalInYobs.max()) ])
    ax4.set_xlabel('% Added')
    ax4.set_ylabel('# cells active in SW')
    #
    ax3 = plt.subplot2grid((2,3),(1,1)) 
    imCB1 = ax3.hist2d(numCellsCorrectInYobs/numCellsTotalInYobs, numCellsTotalInYobs, bins = [ np.linspace(0,1,11), np.linspace(1,numCellsTotalInYobs.max(),numCellsTotalInYobs.max()) ])
    ax3.set_xlabel('% Correct')
    #
    ax5 = plt.subplot2grid((2,3),(1,2)) 
    imCB3 = ax5.hist2d(numCellsDroppedInYobs/numCellsTotalInYobs, numCellsTotalInYobs, bins = [ np.linspace(0,1,11), np.linspace(1,numCellsTotalInYobs.max(),numCellsTotalInYobs.max()) ])
    ax5.set_xlabel('% Dropped')
    #
    imCB = imCB1[3]
    imCBmax = imCB1[0].max()
    if imCB2[0].max() > imCBmax:
        imCBmax = imCB2[0].max()
        imCB = imCB2[3]
    if imCB3[0].max() > imCBmax:
        imCBmax = imCB3[0].max()
        imCB = imCB3[3] 
    #
    cax = g.add_axes([0.93, 0.1, 0.02, 0.3])
    CB = g.colorbar(imCB, cax=cax)
    CB.set_label( str('counts / ' + str(numSamps) + ' observations') )
    #
    plt.tight_layout()
    #
    if not os.path.exists(plt_save_dir):
        os.makedirs(plt_save_dir)
    plt.savefig( str(plt_save_dir + 'Compare_CellsModeled_InferVsTrueVsObs_InferSamples_Params' + params_init  + params_init_param + '_Samples' + str(numSamps) + '.' + figSaveFileType  ) )
    plt.close()







def plot_CA_inference_temporal_performance(Z_inferred_list, Z_list, smp_EM, M_mod, M, numSWs, params_init, params_init_param, r, plt_save_dir, fname_save_tag, figSaveFileType):


    # Active cell assemblies modulo M_sml the ones that cant possibly match
    M_sml = np.min([ M, M_mod ]) # use rc.Zset_modulo_Ms
    M_big = np.max([ M, M_mod ])

    num_samps = len(smp_EM)
    CA_inf_vs_T_Correct = np.zeros((num_samps,M_big))
    CA_inf_vs_T_Added = np.zeros((num_samps,M_big))
    CA_inf_vs_T_Missed = np.zeros((num_samps,M_big))
    CA_true_vs_T = np.zeros((num_samps,M))
    CA_inf_vs_T = np.zeros((num_samps,M_mod))

    # First time point
    i=0
    #
    Z_GT, ZoutG = rc.Zset_modulo_Ms( list(Z_list[smp_EM[i]]), M_sml )
    Z_Mod, ZoutM = rc.Zset_modulo_Ms( list(Z_inferred_list[i]), M_sml )
    #print(i, M_sml,'   //    ', Z_list[i], Z_GT,'   //    ', list(Z_inferred_list[i]), Z_Mod, '   //    ', ZoutG, ZoutM, set.union(ZoutG, ZoutM) )
    #
    CA_inf_vs_T_Correct[i, list(set.intersection(Z_Mod, Z_GT)) ]+=1 # correctly inferred and active.
    CA_inf_vs_T_Added[i,   list(set.difference(Z_Mod, Z_GT)) ]+=1   # incorrectly added in zHyp
    CA_inf_vs_T_Missed[i,  list(set.difference(Z_GT, Z_Mod)) ]+=1   # missed it from zTrue
    CA_true_vs_T[i, list(Z_list[smp_EM[i]]) ]+=1                                 # active in ground truth
    CA_inf_vs_T[i, list(Z_inferred_list[i]) ]+=1   
    #
    # #
    #
    # All following time points.
    for i in range(1,num_samps):
        #
        Z_GT, ZoutG = rc.Zset_modulo_Ms( list(Z_list[smp_EM[i]]), M_sml )
        Z_Mod, ZoutM = rc.Zset_modulo_Ms( list(Z_inferred_list[i]), M_sml )
        #
        CA_inf_vs_T_Correct[i,:] = CA_inf_vs_T_Correct[i-1,:]
        CA_inf_vs_T_Added[i,:] = CA_inf_vs_T_Added[i-1,:]
        CA_inf_vs_T_Missed[i,:] = CA_inf_vs_T_Missed[i-1,:]
        CA_true_vs_T[i,:] = CA_true_vs_T[i-1,:]
        CA_inf_vs_T[i,:] = CA_inf_vs_T[i-1,:]
        #
        CA_inf_vs_T_Correct[i,  list(set.intersection( Z_Mod, Z_GT)) ]+=1   # correctly inferred and active.
        CA_inf_vs_T_Added[i,    list(set.difference( Z_Mod, Z_GT)) ]+=1     # incorrectly added in zHyp
        CA_inf_vs_T_Missed[i,   list(set.difference( Z_GT, Z_Mod)) ]+=1     # missed it from zTrue
        CA_true_vs_T[i, list(Z_list[smp_EM[i]]) ]+=1                                     # active in ground truth
        CA_inf_vs_T[i, list(Z_inferred_list[i]) ]+=1                                     # active in inferred z vector
    #
    plt.rc('font', weight='bold', size=24)
    f,ax = plt.subplots(2,2, figsize=(20,10))
    for i in range(M_sml):
        ax[0][0].plot( CA_inf_vs_T_Correct[:,i] - CA_inf_vs_T_Added[:,i] - CA_inf_vs_T_Missed[:,i], linewidth=4)
        ax[1][0].plot(CA_true_vs_T[:,i], linewidth=4, label=i )
        ax[0][1].plot( CA_inf_vs_T_Correct[:,i] , linewidth=4)
        ax[1][1].plot(-CA_inf_vs_T_Added[:,i] -CA_inf_vs_T_Missed[:,i], linewidth=4, label=i )
    #
    ax[0][0].plot( CA_inf_vs_T_Correct[:,M_sml:].sum(axis=1) - CA_inf_vs_T_Added[:,M_sml:].sum(axis=1) - CA_inf_vs_T_Missed[:,M_sml:].sum(axis=1), linewidth=4,label='all extras')
    ax[0][0].plot( CA_true_vs_T[:,M_sml:].sum(axis=1), linewidth=4,label='all extras')
    ax[0][1].text( 0,0.95*CA_inf_vs_T_Correct.max(), str('#CAs Activated='+str( np.max([M_mod,M]) - (np.where(CA_inf_vs_T_Correct[-1,:]==0)[0]).size )),fontsize=16 )
    ax[0][1].text( 0,0.85*CA_inf_vs_T_Correct.max(), str('M True='+str(M)),fontsize=16 )
    ax[0][1].text( 0,0.75*CA_inf_vs_T_Correct.max(), str('M Model='+str(M_mod)),fontsize=16 ) 

    # #
    # for i in range(M,M_mod):
    #         ax[0][0].plot( CA_inf_vs_T_Correct[:,i] - CA_inf_vs_T_Added[:,i] - CA_inf_vs_T_Missed[:,i], 'k--', linewidth=2)
    #         ax[1][0].plot(CA_true_vs_T[:,i])
    #         ax[1][0].plot(CA_inf_vs_T_Correct[:,i])
    #         ax[1][1].plot(-CA_inf_vs_T_Added[:,i] -CA_inf_vs_T_Missed[:,i])




    plt.suptitle('Inference vs. Time in EM Algorithm')
    #ax[0].set_xlabel('EM Iteration')
    ax[0][0].grid()
    ax[0][0].set_ylabel('<-- #Add/Drop V. #Correct -->')
    #
    ax[1][0].set_xlabel('EM Iteration')
    ax[1][0].grid()
    ax[1][0].set_ylabel('times active')
    #
    ax[0][1].grid()
    ax[0][1].set_ylabel('#Correct')
    #
    ax[1][1].set_xlabel('EM Iteration')
    ax[1][1].grid()
    ax[1][1].set_ylabel('#Add+Drop')
    #
    # if M > 20:
    #     ax[1][0].legend(fontsize=4,title='Cell Assemblies',loc='upper left')
    # elif M > 10:
    #     ax[1][0].legend(fontsize=8,title='Cell Assemblies',loc='upper left') 
    # else:
    #     ax[1][0].legend(fontsize=12,title='Cell Assemblies',loc='upper left')     
    plt.tight_layout()
    #
    if not os.path.exists( str(plt_save_dir+'InferencePerformance/') ):
        os.makedirs( str(plt_save_dir+'InferencePerformance/') )
    plt.savefig( str(plt_save_dir + 'InferencePerformance/' + 'InferenceTemporal_' + fname_save_tag + '.' + figSaveFileType ) )
    plt.close()
    # plt.show()      
    

def visualize_translation_of_CAs( A, Atag, B, Btag, translate, translate2, dot_prod, dot_prod2, 
                        trans_preShuff, ind, Perm, dropWarn, numSamps, r, plt_save_dir, fname_tag ):



    M_a = A.shape[1]    # number of columns / CA's in matrix A
    M_b = B.shape[1]    # number of columns / CA's in matrix B 
    N = A.shape[0]      # number of columns / CA's in matrix A


    if False:
        print(' ')
        print('dot_prod : ',dot_prod.round(2))
        print(' ')
        print('trans_preShuff : ',trans_preShuff)
        print(' ')
        print('reg : ',np.arange(M_a))
        print(' ')
        print('translate : ',translate)
        print(' ')
        print('ind : ',ind)



    if np.max([M_a,M_b]) > 20:
      fsize=6
    elif np.max([M_a,M_b]):
      fsize=8
    else:
      fsize=12  
    

    plt.figure( figsize=(20,10) ) # size units in inches
    plt.rc('font', weight='bold', size=12)
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    ax2 = plt.subplot2grid((2,2),(0,1)) 
    ax2.imshow(Perm)#, vmin=0, vmax=1)
    ax2.set_title('orig cosSim $ \\frac{(A.B)}{ \Vert A \Vert \Vert B \Vert } $')
    ax2.set_xticks(np.arange(M_b))
    ax2.set_xticklabels(np.arange(M_b),fontsize=fsize,rotation=90)
    ax2.set_yticks(np.arange(M_a))
    ax2.set_yticklabels(np.arange(M_a),fontsize=fsize)
    ax2.set_xlabel('column in orig. B')
    ax2.set_ylabel('column in orig. A')
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    ax3 = plt.subplot2grid((2,2),(1,2)) 
    ax3.imshow(Perm[:,trans_preShuff])#, vmin=0, vmax=1) # 
    ax3.plot( [0,np.min([M_a,M_b])-1],[0,np.min([M_a,M_b])-1],'r--', alpha=0.5)
    ax3.set_title('cosSim w/ permutation of columns in B')
    ax3.set_xticks(np.arange(M_b))
    ax3.set_xticklabels(trans_preShuff,fontsize=fsize,rotation=90)#trans_preShuff
    ax3.set_yticks(np.arange(M_a))
    ax3.set_yticklabels(np.arange(M_a),fontsize=fsize)
    ax3.set_xlabel('column in B by reordered by cosSim to orig. A')
    ax3.set_ylabel('column in orig. A')
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    # ax0 = plt.subplot2grid((2,3),(0,0))
    # ax0.imshow(A)
    # ax0.set_title(str('orig. A: '+ Atag) )
    # #ax0.set_xlabel('CAs - $\\vec{z}$')
    # ax0.set_ylabel('cells - $\\vec{y}$')
    # ax0.set_xticks(np.arange(M_a))
    # ax0.set_xticklabels(np.arange(M_a),fontsize=fsize,rotation=90)
    # ax0.yaxis.grid()
    # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # #
    # ax4 =  plt.subplot2grid((2,3),(1,0)) 
    # ax4.imshow(B) # [:,translate[:M_b]]  <-- the translate here is good too!
    # ax4.set_title(str('orig. B: '+ Btag))
    # ax4.set_xlabel('CAs - $\\vec{z}$')
    # ax4.set_ylabel('cells - $\\vec{y}$')
    # ax4.set_xticks(np.arange(M_b))
    # ax4.set_xticklabels(np.arange(M_b),fontsize=fsize,rotation=90)
    # ax4.yaxis.grid()
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    ax1 = plt.subplot2grid((3,2),(0,2))
    ax1.imshow(A[:,ind],aspect='auto') #[:,ind]
    ax1.plot( [np.min([M_a,M_b])-1+0.5,np.min([M_a,M_b])-1+0.5],[0,N-1],'w--')
    ax1.set_xticks(np.arange(M_a))
    ax1.set_xticklabels(ind,fontsize=fsize,rotation=90)
    ax1.yaxis.grid()
    ax1.set_ylabel('cells - $\\vec{y}$')
    ax1.set_title( str('A: '+Atag+' reordered by cosSim w/ B') )
    # ax1.set_xlabel('z')
    # ax1.set_ylabel('y')
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    ax5 = plt.subplot2grid((3,2),(1,0))
    ax5.imshow(B[:, translate],aspect='auto') #
    ax5.plot( [np.min([M_a,M_b])-1+0.5,np.min([M_a,M_b])-1+0.5],[0,N-1],'w--')
    ax5.set_xticks(np.arange(M_a))
    ax5.set_xticklabels(translate,fontsize=fsize,rotation=90) # 
    ax5.yaxis.grid()
    ax5.set_ylabel('cells - $\\vec{y}$')
    ax5.set_title( str('B : '+Btag+' reordered by cosSim w/ A') )
    # ax5.set_xlabel('z')
    # ax5.set_ylabel('y')
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    ax7 = plt.subplot2grid((3,2),(2,0)) #plt.subplot2grid((3,3),(2,0)) 
    ax7.plot( [np.min([M_a,M_b])-1+0.5,np.min([M_a,M_b])-1+0.5],[0,1],'k--')
    ax7.bar(np.arange(dot_prod.size),dot_prod,color='blue',alpha=0.5)
    for i in range(len(ind)):
        ax7.text(i,dot_prod[i],translate[i],horizontalalignment='left',color='blue',fontsize=fsize,rotation=90)
    ax7.bar(np.arange(dot_prod2.size),dot_prod2,color='red',alpha=0.5)
    for i in range(len(ind)):
        ax7.text(i,dot_prod2[i],translate2[i],horizontalalignment='right',color='red',fontsize=fsize,rotation=90)
    #ax7.set_title('similarity')
    ax7.text(np.max(M_a)-2,0.85, str('$\mu$='+str(dot_prod.mean().round(3))), color='blue', horizontalalignment='right',verticalalignment='top')  
    ax7.text(np.max(M_a),0.95,dropWarn,horizontalalignment='right',verticalalignment='top')  
    ax7.set_xlabel('Column in orig. A (column in orig. B above bar)')
    ax7.set_xticks(np.arange(M_a))
    ax7.set_xticklabels(ind,fontsize=fsize,rotation=90)
    ax7.yaxis.grid()
    ax7.set_ylabel(' $ \\frac{(A.B)}{ \Vert A \Vert \Vert B \Vert } $')    
    ax7.set_ylim(0,1)
    ax7.set_xlim(0,M_a)
    #
    plt.suptitle( str('Finding Cell Assemblies in Matrix B (' + Btag + ') that best fit Cell Assemblies in Matrix A (' + Atag + ')'), fontsize=18)
    plt.tight_layout()  
    #
    print(str(plt_save_dir + 'translatePermute/' + 'translatePermute_' + Atag[:5] + 'to' + Btag[:5] + '_' + fname_tag + '.' + figSaveFileType))
    #
    if not os.path.exists( str(plt_save_dir+'translatePermute/') ):
        os.makedirs( str(plt_save_dir+'translatePermute/') )
    plt.savefig( str(plt_save_dir + 'translatePermute/' + 'translatePermute_' + Atag[:5] + 'to' + Btag[:5] + '_' + fname_tag + '.' + figSaveFileType ) ) 
    plt.close() 











def visualize_matchModels_cosSim( A, Atag, B, Btag, ind, cos_sim, len_dif, cosSimMat, lenDifMat, numSamps, r, plt_save_dir, fname_tag, figSaveFileType ):

    M_a = A.shape[1]    # number of columns / CA's in matrix A
    M_b = B.shape[1]    # number of columns / CA's in matrix B 
    N = A.shape[0]      # number of columns / CA's in matrix A


    if len(cos_sim)==2:
        csSrt   = np.nanmean(cos_sim[0])
    else:
        csSrt   = np.nanmean(cos_sim)


    # Null Models: CosSim & LenDif of unordered matrices.
    
    csNM1   = np.nanmean( np.diag(cosSimMat) ) 
    csNM2   = np.nanmean( cosSimMat ) 
    ldSrt   = np.nanmean(len_dif[0])
    ldNM1   = np.nanmean( np.diag(lenDifMat) )
    ldNM2   = np.nanmean( lenDifMat )





    if False:

        print('cosSim Sort: ', np.nanmean(cosSim[0]).round(2) )
        print('cosSim Null: ', csNM1.round(2) , csNM2.round(2) )
        print('lenDif Sort: ', np.nanmean(lenDif[0]).round(2) )
        print('cosSim Null: ', ldNM1.round(2) , ldNM2.round(2) )

        print(' ')
        print('cos_sim : ',cos_sim[0].round(2))
        print(' ')
        print('len_dif : ',len_dif[0].round(2))
        print(' ')
        print('ind_B : ',ind[0])
        print(' ')
        print('reg : ',np.arange(M_a))
        print(' ')
        print('ind_A : ',ind[1])
        print(' ')



    if np.max([M_a,M_b]) > 20:
      fsize=5
    elif np.max([M_a,M_b]):
      fsize=8
    else:
      fsize=12  
    

    f = plt.figure( figsize=(15,10) ) # size units in inches
    plt.rc('font', weight='bold', size=12)
    plt.rc('text', usetex=False) # False because some weird dvi file not found problem ???
    #plt.rcParams['text.latex.preamble'] = [r'\boldmath']
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    ax2 = plt.subplot2grid((2,2),(0,0)) 
    ax2.imshow(cosSimMat, vmin=0, vmax=1)
    ax2.plot( [0,np.min([M_a,M_b])-1], [0,np.min([M_a,M_b])-1], 'r--', alpha=0.5)
    ax2.set_title('Unmatched cosSim')
    ax2.set_xticks(np.arange(M_a))
    ax2.set_xticklabels(np.arange(M_a),fontsize=fsize,rotation=90)
    ax2.set_yticks(np.arange(M_b))
    ax2.set_yticklabels(np.arange(M_b),fontsize=fsize)
    ax2.set_xlabel( str('column in orig. '+Atag) )
    ax2.set_ylabel( str('column in orig. '+Btag) )
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    ax3 = plt.subplot2grid((2,2),(1,0)) 
    ax3.imshow(cosSimMat[np.ix_(ind[0],ind[1])], vmin=0, vmax=1) # 
    ax3.plot( [0,np.min([M_a,M_b])-1], [0,np.min([M_a,M_b])-1], 'r--', alpha=0.5)
    ax3.plot( [np.min([M_a,M_b])-1+0.5, np.min([M_a,M_b])-1+0.5], [0, M_b-1], 'w--', alpha=0.5)
    ax3.plot( [0, M_b-1], [np.min([M_a,M_b])-1+0.5, np.min([M_a,M_b])-1+0.5], 'w--', alpha=0.5)
    ax3.set_title( 'Matched cosSim' )
    ax3.set_xticks( np.arange( M_b ) ) 
    ax3.set_xticklabels(ind[1],fontsize=fsize,rotation=90)#trans_preShuff
    ax3.set_yticks( np.arange( M_b ) )
    ax3.set_yticklabels(ind[0],fontsize=fsize)
    ax3.set_xlabel( str('column in '+Atag+' (sorted by cosSim size)') )
    ax3.set_ylabel( str('column in '+Btag+' (reordered to match)') )
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    # ax0 = plt.subplot2grid((2,3),(0,0))
    # ax0.imshow(A, vmin=0, vmax=1)
    # ax0.set_title(str('orig. '+ Atag) )
    # #ax0.set_xlabel(r'CAs - $\vec{z}$')
    # ax0.set_ylabel(r'cells - $\vec{y}$')
    # ax0.set_xticks(np.arange(M_a))
    # ax0.set_xticklabels(np.arange(M_a),fontsize=fsize,rotation=90)
    # ax0.yaxis.grid()
    # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # #
    # ax4 =  plt.subplot2grid((2,3),(1,0)) 
    # ax4.imshow(B, vmin=0, vmax=1) # [:,translate[:M_b]]  <-- the translate here is good too!
    # ax4.set_title(str('orig. '+ Btag))
    # ax4.set_xlabel(r'CAs - $\vec{z}$')
    # ax4.set_ylabel(r'cells - $\vec{y}$')
    # ax4.set_xticks(np.arange(M_b))
    # ax4.set_xticklabels(np.arange(M_b),fontsize=fsize,rotation=90)
    # ax4.yaxis.grid()
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    ax1 = plt.subplot2grid((3,2),(0,1))
    ax1.imshow(A[:,ind[1]],aspect='auto', vmin=0, vmax=1) #[:,ind]
    ax1.plot( [np.min([M_a,M_b])-1+0.5, np.min([M_a,M_b])-1+0.5],[0,N-1], 'w--', alpha=0.5)
    ax1.set_xticks( np.arange( M_b ) )
    ax1.set_xticklabels(ind[1],fontsize=fsize,rotation=90)
    ax1.yaxis.grid()
    ax1.set_ylabel(r'cells - $\vec{y}$')
    ax1.set_title( str('Matched '+Atag) )
    # ax1.set_xlabel('z')
    # ax1.set_ylabel('y')
    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    ax5 = plt.subplot2grid((3,2),(1,1))
    s = ax5.imshow(B[:, ind[0]],aspect='auto', vmin=0, vmax=1) #
    ax5.plot( [np.min([M_a,M_b])-1+0.5, np.min([M_a,M_b])-1+0.5], [0,N-1], 'w--', alpha=0.5)
    ax5.set_xticks(np.arange(M_b))
    ax5.set_xticklabels(ind[0],fontsize=fsize,rotation=90) # 
    ax5.yaxis.grid()
    ax5.set_ylabel(r'cells - $\vec{y}$')
    ax5.set_title( str('Matched '+Btag) )
    # ax5.set_xlabel('z')
    # ax5.set_ylabel('y')
    #
    cbar_ax = f.add_axes([0.17, 0.03, 0.25, 0.02])
    cb1=f.colorbar(s, cax=cbar_ax, orientation='horizontal')



    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    if len(cos_sim)==2:
        mix_dst = cos_sim # This is for our reordering. Not hungarian method. Add a second dimension to avoid errors (a cludge for now).
    else:
        mix_dst = np.vstack( (cos_sim, np.zeros_like(cos_sim)) )
    #
    # mix_dst = cos_sim[0]*(1-len_dif[0])
    # mix_dst2 = cos_sim[1]*(1-len_dif[1])
    #
    ax7 = plt.subplot2grid((3,2),(2,1)) #plt.subplot2grid((3,3),(2,0)) 
    ax7.plot( [np.min([M_a,M_b])-1+0.5,np.min([M_a,M_b])-1+0.5],[0,1],'k--')
    ax7.bar(np.arange(mix_dst[0].size),mix_dst[0],color='blue',alpha=0.5) #,label=str(r'$1^{st}, \mu$='+str(cos_sim[0].mean().round(3)) ) )
    for i in range(len(ind[1])):
        ax7.text(i,mix_dst[0][i],ind[1][i],ha='left',va='bottom',color='blue',fontsize=fsize,rotation=90)
    # #
    # nmSort = np.argsort(csNM)[::-1]
    # ax7.bar(np.arange(csNM.size),csNM[nmSort],color='black',alpha=0.3) #,label=str(r'$1^{st}, \mu$='+str(cos_sim[0].mean().round(3)) ) )
    # ax7.scatter( np.arange(M_a), inferNorm[nmSort], s=10, color='red', marker='x')
    # #
    if len(ind)==3: # Do this only for our reordering, not for Hungarian method.
        ax7.bar(np.arange(mix_dst[1].size),mix_dst[1],color='red',alpha=0.3,label=r'$2^{nd}$ best CA')
        for i in range(len(ind[1])):
            ax7.text(i,mix_dst[1][i],ind[2][i],ha='right',va='top',color='black',fontsize=fsize,rotation=90)
        #
    #ax7.set_title('similarity')
    #ax7.scatter( np.arange(M_b), len_dif[0], s=10, c='k', marker='x',label= str( r'$1-\frac{\vec{A}-\vec{B}}{\vec{A}+\vec{B}} \mu$='+str(len_dif[0].mean().round(3)) ) )
    #
    multicolor_axis_label(ax7,(str('CAs = [ '+Btag+' '), str('best '+Atag+' '), str(' ]')),('k','b','k'),axis='x',size=12,weight='bold', xoffset=-0.45) # ,ha='center')#
    ax7.set_xticks(np.arange( np.max([M_a,M_b])))
    ax7.set_xticklabels( np.array(ind).T, fontsize=fsize,rotation=90) # ,str( str(ind[0])+' '+str(ind[1])+' '+str(ind[1]) )
    ax7.yaxis.grid()
    ax7.set_ylabel(r'cosSim = $ \frac{(A.B)}{ \Vert A \Vert \Vert B \Vert } $')    
    ax7.set_ylim( 0,1 )
    ax7.set_xlim( 0,M_b )
    ax7.legend(fontsize=fsize)
    #
    plt.tight_layout()
    plt.suptitle( str('Matching Cell Assemblies in '+Atag+' (M='+str(M_a)+') and '+Btag+' (M='+str(M_b)+') \n' \
        + r' (sort,null): $\mu_{CS}$ = ('+ str(csSrt.round(2))+','+str(csNM1.round(2))+')'), fontsize=18) # +r'$\mu_{LD}$ = ('+ str(ldSrt.round(2))+','+str(ldNM1.round(2))+')'

    print(str(plt_save_dir + 'translatePermute/' + 'translatePermute_' + Atag[:5] + '_' + Btag[:5] + '_' + fname_tag + '.' + figSaveFileType))
    #
    if not os.path.exists( str(plt_save_dir+'translatePermute/') ):
        os.makedirs( str(plt_save_dir+'translatePermute/') )
    plt.savefig( str(plt_save_dir + 'translatePermute/' + 'translatePermute_' + Atag[:5] + '_' + Btag[:5] + '_' + fname_tag + '.' + figSaveFileType ) ) 
    plt.close() 
















def multicolor_axis_label(ax,list_of_strings,list_of_colors,axis='x',xoffset=-0.1,yoffset=-0.1,anchorpad=0,**kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    #
    # Got from: https://stackoverflow.com/questions/33159134/matplotlib-y-axis-label-with-multiple-colors
    #
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.1, xoffset),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw)) 
                     for text,color in zip(list_of_strings[::-1],list_of_colors) ]
        ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(yoffset, 0.1), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)









def plot_learned_model(Pi, Pia, Q, numCAsUB, numCAsLB, numCellsUB, numCellsLB, Zassem_hist_Infer,\
                        Ycell_hist_Infer, yInferSampled_EM, TH_bounds, maxNumCells, maxNumCAs, maxPi, nY_Samps, nY_Infer, nZ_Infer, \
                        sampAt, samps, EM_figs_dir, plt_save_tag, plt_title, figSaveFileType):

    # NOTE: CLEAN UP THIS FUNCTION SO WE ARE PASSING IN ri, ria and q.
    # Pia = 
    # Pi = 
    # Q = 

    N = Pia.shape[0]
    M = Pia.shape[1]
    activeZs  = np.where(Zassem_hist_Infer!=0)[0]
    #
    if activeZs.size==0:
        activeZs = np.array([1])
    #
    # try:
    colors = plt.cm.jet_r(np.linspace(0,1,activeZs.max()+1))
    # except:
    #     print('Something weird inside plot_learned_model function.')
    #     #return

    nYT = 4

    ## Imshow Pia Cell vs. Cell Assembly matrix.
    #plt.rc('font', weight='bold', size=16)
    f=plt.figure( figsize=(15,10) )
    ax0 = plt.subplot2grid( (5,5), (0,0), colspan=3, rowspan=4 ) 
    im0=ax0.imshow(Pia, vmin=0,vmax=1, aspect='auto', cmap='viridis')
    #for i in activeZs:
    #    ax0.scatter(i, N, marker='s', s=30, color=colors[i])
    ax0.set_xlim(0-0.5,M+0.5)
    ax0.xaxis.set_ticklabels([])
    ax0.tick_params(axis='both', labelsize=14)
    ax0.set_ylim(0-0.5,N+0.5)
    ax0.set_xticks( np.linspace(0,M,11).astype(int) )
    ax0.set_yticks( np.linspace(0,N,11).astype(int) )
    ax0.invert_yaxis()
    ax0.set_ylabel('Cell id')
    ax0.set_title( str('Pia'),fontsize=18 )
    ax0.grid()
    cax0 = f.add_axes([0.65, 0.1, 0.25, 0.05])
    f.colorbar(im0, cax=cax0, orientation='horizontal')
    cax0.set_label('Pia vals')

    #

    ## HORIZONTAL BOTTOM PLOT: Scatter plot numCells per cell assembly and numTimes each CA is inferred. 

    ax1 = plt.subplot2grid( (5,5), (4,0), colspan=3)
    indL1 = np.where(numCellsLB==0)[0].astype(int)
    indL2 = np.where(numCellsLB>0)[0].astype(int)
    ax1.scatter( range(indL2.size), np.log2(numCellsLB[indL2]), marker='^', s=30, color='green', alpha=0.5, label=str('$\Theta_{LB}='+str(TH_bounds.max())+'$') )
    indU1 = np.where(numCellsUB==0)[0].astype(int)
    indU2 = np.where(numCellsUB>0)[0].astype(int)
    ax1.scatter( range(indU2.size), np.log2(numCellsUB[indU2]), marker='v', s=30, color='blue', alpha=0.5, label=str('$\Theta_{UB}='+'0.5'+'$') ) # str(TH_bounds.min())
    ax1.set_ylabel('# Cells per CA', color='blue')
    ax1.set_xlabel('Cell Assembly id')
    ytix = np.unique( np.concatenate([ numCellsLB[indL2], numCellsUB[indU2]  ]) ) 
    ax1.set_yticks( np.log2(ytix) )
    ax1.set_yticklabels(ytix)
                                    
    ax1.set_xticks( np.linspace(0,M,11).astype(int) )
    ax1.set_xlim(0-0.5,M+0.5)
    ax1.invert_yaxis()
    ax1.grid()
    #ax1.text(M, 0.5, '$log_{10}$ prob', color='red', horizontalalignment='right', fontsize=16)  # bbox={'facecolor': 'red'},)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=10)
    ax1.tick_params(axis='both', labelsize=10)
    ax1.grid(color='blue',linestyle='--',linewidth=0.5)
    ax1.legend(bbox_to_anchor=(1.25, 0.8), fontsize=10)
    #
    ax1b = ax1.twinx()
    ax1b.scatter(   np.where(Zassem_hist_Infer>0)[0].astype(int), np.log10(Zassem_hist_Infer[np.where(Zassem_hist_Infer>0)[0]]/sampAt), \
                            s=50, color='red', marker='x', alpha=0.5 )
    ax1b.grid(color='red',linestyle='--',linewidth=0.5)
    ax1b.invert_yaxis()
    ax1b.tick_params(axis='y', labelcolor='red', labelsize=10)
    try:
        ax1b.set_yticks( np.linspace(np.log10(Zassem_hist_Infer[np.where(Zassem_hist_Infer>0)[0]]/sampAt).max(), \
                                np.log10(Zassem_hist_Infer[np.where(Zassem_hist_Infer>0)[0]]/sampAt).min()  ,nYT) )
        ax1b.set_yticklabels( np.round( np.logspace(np.max(np.log10(Zassem_hist_Infer[np.where(Zassem_hist_Infer>0)[0]])), \
                                np.min(np.log10(Zassem_hist_Infer[np.where(Zassem_hist_Infer>0)[0]])) ,nYT)).astype(int) )
    except:
        print('What the fuck ever.')
    #
    # #
    #
    ## VERTICAL INSIDE-RIGHT PLOT: Scatter plot numCAs per cell and numTimes each cell is inferred. 
    ax2 = plt.subplot2grid( (5,5), (0,3), rowspan=4 )
    ax2.scatter( np.log10(Ycell_hist_Infer[np.where(Ycell_hist_Infer>0)[0]]/sampAt), np.where(Ycell_hist_Infer>0)[0].astype(int), \
                            s=50, color='red', marker='x', alpha=0.5, label=str('infrd') )
    ax2.scatter( np.log10(yInferSampled_EM[np.where(yInferSampled_EM>0)[0]]/samps), np.where(yInferSampled_EM>0)[0].astype(int), \
                            s=30, color='red', marker='+', alpha=0.5, label=str('sampd') )
    #ax2.set_xlim(0, np.log10(np.hstack([Ycell_hist_Infer[:-1]/sampAt, yInferSampled_EM[:-1]/samps])).max() )
    ax2.tick_params(axis='x', labelcolor='red', labelsize=10)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.set_xlabel('inference prob', color='red')
    #ax2.invert_yaxis()
    ax2.yaxis.set_ticklabels([])
    ax2.set_ylim(0-0.5,N+0.5)
    ax2.set_yticks( np.linspace(0,N,11).astype(int) )
    try:
        ax2.set_xticks( np.linspace(np.log10(Ycell_hist_Infer[np.where(Ycell_hist_Infer>0)[0]]/sampAt).max(), \
                                np.log10(Ycell_hist_Infer[np.where(Ycell_hist_Infer>0)[0]]/sampAt).min() ,nYT) )
        ax2.set_xticklabels( np.round( np.logspace(np.max(np.log10(Ycell_hist_Infer[np.where(Ycell_hist_Infer>0)[0]])), \
                                np.min(np.log10(Ycell_hist_Infer[np.where(Ycell_hist_Infer>0)[0]])) ,nYT)).astype(int) )
    except:
        print('What the fuck ever.')
    

    #ax2.set_xticklabels( np.round(np.logspace(0, np.log10(Ycell_hist_Infer[np.where(Ycell_hist_Infer>0)[0]]/sampAt) , nYT) ).astype(int) )
    ax2.grid(color='red',linestyle='--',linewidth=0.5)
    ax2.legend(bbox_to_anchor=(1.15, -0.09), fontsize=10)
    #
    ax2b = ax2.twiny()
    ax2b.scatter( numCAsUB[np.where(numCAsUB>0)].astype(int), np.where(numCAsUB>0)[0].astype(int), \
                            marker='>', s=30, color='blue', alpha=0.5) #, label=str('$\Theta_{UB}='+str(TH_bounds.min())+'$') )
    ax2b.scatter( numCAsLB[np.where(numCAsLB>0)].astype(int), np.where(numCAsLB>0)[0].astype(int), \
                            marker='<', s=30, color='green', alpha=0.5) #, label=str('$\Theta_{LB}='+str(TH_bounds.max())+'$') )
    ax2b.set_xlabel('# CAs per cell', color='blue')
    ax2b.set_xticks( np.arange(maxNumCAs.max()+1) )
    ax2b.tick_params(axis='x', labelcolor='blue', labelsize=10)
    ax2b.invert_yaxis()
    ax2b.grid(color='blue',linestyle='--',linewidth=0.5)
    #ax2b.legend(fontsize=10)
    #
    # #
    #
    ## VERTICAL OUTSIDE-RIGHT PLOT: Scatter plot Pi values for each cell and Q value. 
    ax3 = plt.subplot2grid( (5,5), (0,4), rowspan=4 )
    ax3.scatter( 100*Q, N/2, s=100, color='magenta', marker='d', alpha=0.9, label='Q' )
    ax3.scatter( 100*Pi[np.where(Pi>0)], np.where(Pi>0)[0].astype(int), s=50, color='black', alpha=0.5, label='$P_i$' )
    ax3.set_xlabel('P(Q/$P_i$=1) %')
    ax3.set_ylabel('Cell id')
    ax3.set_yticks( np.linspace(0,N,11).astype(int) )
    ax3.set_ylim(0-0.5,N+0.5)
    ax3.set_xticks(  np.round( 100*np.linspace(0,maxPi ,5)).astype(int) )
    ax3.set_xlim(0,100*maxPi+1)
    #ax3.yaxis.set_ticklabels([])
    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position('top')
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')
    ax3.invert_yaxis()
    ax3.tick_params(axis='both', labelsize=10)
    ax3.grid()
    ax3.legend(bbox_to_anchor=(0.23, -0.09), fontsize=8)
    # ax1b = ax1.twinx()
    #
    # #
    #

    # #
    # ax4a = plt.subplot2grid( (10,10), (8,6), rowspan=2, colspan=2) #, aspect='equal') #, autoscale_on=False )
    # x1=np.histogram(numCAsUB,bins=np.arange(maxNumCAs.max()+2))
    # ax4a.scatter(x1[1][:-1],x1[0]/N, s=100, alpha=0.9, color='blue', marker='^', linewidth=2, label=str('$\Theta_{UB}='+str(TH_bounds.min())+'$') )
    # x2=np.histogram(numCAsLB,bins=np.arange(maxNumCAs.min()+2))
    # ax4a.scatter(x2[1][:-1],x2[0]/N, s=100, alpha=0.6, color='green', marker='v', linewidth=2, label=str('$\Theta_{LB}='+str(TH_bounds.max())+'$') )
    # #ax4a.set_ylim(0,N+1)
    # ax4a.set_yticks( np.round(np.linspace(0,1,5), 1) )
    # ax4a.yaxis.set_ticklabels([])
    # ax4a.set_xlabel('As per C', fontsize=14)
    # ax4a.set_ylabel('counts', fontsize=14)
    # ax4a.set_title('Histograms', fontsize=16)
    # ax4a.grid()
    # ax4a.tick_params(axis='both', labelsize=10)
    # ax4a.set_xticks(np.arange(maxNumCAs.max()))
    # a0,a1 = ax4a.get_xlim()
    # b0,b1 = ax4a.get_ylim()
    # ax4a.set_aspect(abs(a1-a0)/abs(b1-b0))
    # #
    # ax4b = plt.subplot2grid( (10,10), (8,8), rowspan=2, colspan=2) #, aspect='equal') #, autoscale_on=False )
    # y1=np.histogram(numCellsUB,bins=np.arange(maxNumCells.max()+2))
    # ax4b.scatter(y1[1][:-1], y1[0]/M, s=100, alpha=0.9, color='blue', marker='^', linewidth=2, label=str('$\Theta_{UB}='+str(TH_bounds.min())+'$') )
    # y2=np.histogram(numCellsLB,bins=np.arange(maxNumCells.min()+2))
    # ax4b.scatter(y2[1][:-1], y2[0]/M, s=100, alpha=0.6, color='green', marker='v', linewidth=2, label=str('$\Theta_{LB}='+str(TH_bounds.max())+'$') )
    # #ax4b.set_ylim(0,M+1)
    # ax4b.set_yticks( np.round(np.linspace(0,1,5), 1) )
    # ax4b.set_xlabel('Cs per A', fontsize=14)
    # ax4b.yaxis.set_ticklabels([])
    # #ax4b.yaxis.tick_right()
    # ax4b.tick_params(axis='both', labelsize=10)
    # ax4b.set_xticks(np.arange(maxNumCells.max()) )
    # ax4b.grid()
    # a0,a1 = ax4b.get_xlim()
    # b0,b1 = ax4b.get_ylim()
    # ax4b.set_aspect(abs(a1-a0)/abs(b1-b0))
    # #
    # ax4c = plt.subplot2grid( (10,10), (8,4), rowspan=2, colspan=2) #, aspect='equal') #, autoscale_on=False )
    #  # compute histograms from sY_Smp, nY_Inf and nZ_Inf
    # binns = np.arange( np.concatenate([nY_Infer, nY_Samps, nZ_Infer]).min(), np.concatenate([nY_Infer, nY_Samps, nZ_Infer]).max()+2 ) #nY, 
    # nYh_Infer = np.histogram(nY_Infer,bins=binns)
    # nYh_Samps = np.histogram(nY_Samps,bins=binns)
    # nZh_Infer = np.histogram(nZ_Infer,bins=binns) 
    # #
    # ind = np.where(nYh_Samps[0]!=0)[0]
    # ax4c.plot(nYh_Samps[1][ind],    (nYh_Samps[0][ind]/samps), 'go-', linewidth=3, markersize=10, alpha=0.5, label='sampled |y|')
    # ind = np.where(nYh_Infer[0]!=0)[0]
    # ax4c.plot(nYh_Infer[1][ind],    (nYh_Infer[0][ind]/samps), 'ro-', linewidth=3, markersize=10, alpha=0.5, label='inferred |y|')
    # ind = np.where(nZh_Infer[0]!=0)[0]
    # ax4c.plot(nZh_Infer[1][ind],    (nZh_Infer[0][ind]/samps), 'ko--', linewidth=3, markersize=10, alpha=0.5, label='inferred |z|')
    # ax4c.set_xlabel('card | |', fontsize=12)
    # ax4c.set_ylabel('prob', fontsize=12)
    # ax4c.set_yticks( np.round(np.linspace(0,1,5), 1) )
    # ax4c.set_xticks( np.round(np.linspace(0,binns.max(),5)).astype(int) )
    # ax4c.tick_params(axis='both', labelsize=10)
    # ax4c.grid()
    # ax4c.legend(fontsize=10)
    # a0,a1 = ax4c.get_xlim()
    # b0,b1 = ax4c.get_ylim()
    # ax4c.set_aspect(abs(a1-a0)/abs(b1-b0))

    #
    plt.suptitle( plt_title, fontsize=18 )
    plt.tight_layout()
    #
    plt.savefig( str(EM_figs_dir + plt_save_tag + '.' + figSaveFileType ) )
    plt.close() 










# # OLD VERSION. LOOK AT ELLIPSE_RFZ_2AX NOW.
# def ellipse_RFz(STRF_gauss, Pia, A, ax, Bs, AsInB, cols, TH, ZorY, Nsplit=None): 



#     N = STRF_gauss.shape[0]
#     stretch=2
#     ells = [ mpp.Ellipse( (STRF_gauss[i,1],STRF_gauss[i,3]), stretch*STRF_gauss[i,2], stretch*STRF_gauss[i,4], np.rad2deg(2*np.pi-STRF_gauss[i,5])  ) for i in range(N) ] # parameters in STRF_gauss are: [Amplitude, x0, sigmax, y0, sigmay, angle(in rad)]
#     #


#     if not Nsplit:
#         Nsplit=N



#     # If we are plotting cells belonging to CA, then plot edges fully dark with no face color. Because edges are depending on face alpha.
#     if ZorY=='z':
#         j=0 # counter for good colors of ys involved in z. (passing threshold)
#         for i,e2 in enumerate(ells):
#             ax.add_artist(e2)
#             e2.set_clip_box(ax.bbox)
#             e2.set_facecolor('none') 
#             e2.set_alpha(1)
#             e2.set_linestyle('-')
#             #
#             if i in Bs:
#                 e2.set_edgecolor(cols[j])
#                 e2.set_linewidth(3)
#                 j+=1
#             else:
#                 e2.set_edgecolor('lightslategrey')
#                 e2.set_linewidth(1) 
#             #    
#             ax.add_patch(e2)




#     # Plot colored ellipses indicating cells in CA that cell y participates in also.
#     # THIS IS FUCKING BUGGY AND I DONT KNOW WHY. SOMETIMES IT DRAWS ALL COLORED 
#     # ELLIPSES CORRECTLY. SOMETIMES IT JUST LEAVES THEM OUT. I DONT FUCKIN KNOW.
#     if ZorY=='y': 
#         for j in range(N):
#             abc=False
#             print('j=',j)
#             if j<len(AsInB):
#                 print('AsInB[j] = ',AsInB[j])
#                 for k in AsInB[j]:
#                     print('k = ',k)
#                     #if i in AsInB[j]:
#                         #if k in AsInB_already:
#                     nCAs = np.where(np.array(AsInB_already)==k)[0].size
#                     print('AsInB_already=',AsInB_already)
#                     print('nCAs=',nCAs)
#                     e2 = mpp.Ellipse( (STRF_gauss[k,1],STRF_gauss[k,3]), (1-0.2*nCAs)*stretch*STRF_gauss[k,2], (1-0.2*nCAs)*stretch*STRF_gauss[k,4], np.rad2deg(2*np.pi-STRF_gauss[k,5]) )
#                     ax.add_artist(e2)
#                     e2.set_clip_box(ax.bbox)
#                     #
#                     e2.set_alpha(1)
#                     e2.set_linestyle('-')
#                     e2.set_edgecolor(cols[j])
#                     e2.set_linewidth(3)
#                     e2.set_facecolor('none') 
#                     ax.add_patch(e2)
#                     abc=True
#                     AsInB_already.append(k)
#                     print(k,abc)
#             if abc:
#                 j+=1            





#     j=0 # counter for good colors of ys involved in z. (passing threshold)
#     AsInB_already = list()
#     for i,e in enumerate(ells):
#         ax.add_artist(e)
#         e.set_clip_box(ax.bbox)
#         #
#         # (1). Plot all Cell RF's colored red with intensity reflecting membership in a given Cell Assembly (column of Pia).
#         #      And indicate cells that are a strong member of a CA (ie. pass a Threshold in Pia column).
#         if ZorY=='z':
#             e.set_alpha(0.5*Pia[i,A])
#             if i < Nsplit: 
#                 e.set_facecolor('red')
#             else:
#                 e.set_facecolor('blue') 
#         #
#         # (2). Plot all Cell RF's with ALPHA BASED ON FULL SPIKE RATE? Outline cells co-involved in CAs that a given Cell is a member of.
#         #      with colors reflecting the CAs.
#         elif ZorY=='y':             
#             e.set_alpha(1)
#             e.set_edgecolor('lightslategrey')
#             e.set_facecolor('none')  
#             e.set_linewidth(2)
#             e.set_linestyle('-')
#             ax.add_patch(e)
#         else:
#             'This should never happen.'

#         ax.text( STRF_gauss[i,1],STRF_gauss[i,3], i, ha='center', va='center', fontsize=10 )


#     # Plot a colorbar or alpha values matched up with Pia values, indicating our threshold for CA membership
#     xmin = -5
#     xmax = 45
#     ymin = 5
#     ymax = 40
#     cbVal = np.linspace(0,1,5)
#     cbVal = cbVal[1:]
#     #
#     CB = [ mpp.Ellipse( (xmin+i*0.7*(xmax-xmin), ymin+2), 3, 4, 90  ) for i in list( cbVal ) ] # parameters are: [Amplitude, x0, sigmax, y0, sigmay, angle(in rad)]
#     for i,c in enumerate(CB):
#         ax.add_artist(c)
#         c.set_clip_box(ax.bbox)
#         c.set_facecolor('red') 
#         c.set_alpha(0.5*cbVal[i])
#         c.set_linestyle('-')
#         c.set_edgecolor('lightslategrey')
#         c.set_linewidth(1) 
#         ax.add_patch(c)  
#         ax.text( xmin+cbVal[i]*0.7*(xmax-xmin), ymin+2, str(cbVal[i].round(2)).replace('0',''), ha='center', va='center', fontsize=10 )
#     #
#     if N > Nsplit:
#         CB2 = [ mpp.Ellipse( (xmin+i*0.7*(xmax-xmin) + 0.1*(xmax-xmin) , ymin+2), 3, 4, 90  ) for i in list( cbVal ) ] # parameters are: [Amplitude, x0, sigmax, y0, sigmay, angle(in rad)]
#         for i,d in enumerate(CB2):
#             ax.add_artist(d)
#             d.set_clip_box(ax.bbox)
#             d.set_facecolor('blue') 
#             d.set_alpha(0.5*cbVal[i])
#             d.set_linestyle('-')
#             d.set_edgecolor('lightslategrey')
#             d.set_linewidth(1) 
#             ax.add_patch(d)  
#             ax.text( xmin + cbVal[i]*0.7*(xmax-xmin) + 0.1*(xmax-xmin), ymin+2, str(cbVal[i].round(2)).replace('0',''), ha='center', va='center', fontsize=10 )
    
#     ax.text( xmin, ymin+2, str('$P_{ia}=$'), ha='left', va='center', fontsize=10 )   
#     ax.text( xmax, ymin+2, str('$\Theta=$'+str(TH)), ha='right', va='center', fontsize=10 )   
#     ax.plot([xmin,xmax],[ymin+4,ymin+4],'k--') 
#     #
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymin, ymax)
#     ax.invert_yaxis()


#     return ax










def ellipse_RFz_2ax(STRF_gauss, Pia, A, ax, ax2, Bs, AsInB, cols, TH, ZorY, Nsplit=None): 



    N = STRF_gauss.shape[0]
    stretch=2
    ells = [ mpp.Ellipse( (STRF_gauss[i,1],STRF_gauss[i,3]), stretch*STRF_gauss[i,2], stretch*STRF_gauss[i,4], np.rad2deg(2*np.pi-STRF_gauss[i,5])  ) for i in range(N) ] # parameters in STRF_gauss are: [Amplitude, x0, sigmax, y0, sigmay, angle(in rad)]
    #


    if not Nsplit:
        Nsplit=N



    # If we are plotting cells belonging to CA, then plot edges fully dark with no face color. Because edges are depending on face alpha.
    if ZorY=='z':
        j=0 # counter for good colors of ys involved in z. (passing threshold)
        for i,e2 in enumerate(ells):
            if i < Nsplit:
                ax.add_artist(e2)
                e2.set_clip_box(ax.bbox)
            else:
                ax2.add_artist(e2)
                e2.set_clip_box(ax2.bbox)
            e2.set_facecolor('none') 
            e2.set_alpha(1)
            e2.set_linestyle('-')
            #
            if i in Bs:
                e2.set_edgecolor(cols[j])
                e2.set_linewidth(3)
                j+=1
            else:
                e2.set_edgecolor('lightslategrey')
                e2.set_linewidth(1) 
            #    
            if i < Nsplit:
                ax.add_patch(e2)
            else:
                ax2.add_patch(e2)




    # Plot colored ellipses indicating cells in CA that cell y participates in also.
    # THIS IS FUCKING BUGGY AND I DONT KNOW WHY. SOMETIMES IT DRAWS ALL COLORED 
    # ELLIPSES CORRECTLY. SOMETIMES IT JUST LEAVES THEM OUT. I DONT FUCKIN KNOW.
    if ZorY=='y': 
        for j in range(N):
            abc=False
            print('j=',j)
            if j<len(AsInB):
                print('AsInB[j] = ',AsInB[j])
                for k in AsInB[j]:
                    print('k = ',k)
                    #if i in AsInB[j]:
                        #if k in AsInB_already:
                    nCAs = np.where(np.array(AsInB_already)==k)[0].size
                    print('AsInB_already=',AsInB_already)
                    print('nCAs=',nCAs)
                    e2 = mpp.Ellipse( (STRF_gauss[k,1],STRF_gauss[k,3]), (1-0.2*nCAs)*stretch*STRF_gauss[k,2], (1-0.2*nCAs)*stretch*STRF_gauss[k,4], np.rad2deg(2*np.pi-STRF_gauss[k,5]) )
                    ax.add_artist(e2)
                    e2.set_clip_box(ax.bbox)
                    #
                    e2.set_alpha(1)
                    e2.set_linestyle('-')
                    e2.set_edgecolor(cols[j])
                    e2.set_linewidth(3)
                    e2.set_facecolor('none') 
                    ax.add_patch(e2)
                    abc=True
                    AsInB_already.append(k)
                    print(k,abc)
            if abc:
                j+=1            





    j=0 # counter for good colors of ys involved in z. (passing threshold)
    AsInB_already = list()
    for i,e in enumerate(ells):


        if i < Nsplit:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
        else:
            ax2.add_artist(e)
            e.set_clip_box(ax2.bbox)


        #
        # (1). Plot all Cell RF's colored red with intensity reflecting membership in a given Cell Assembly (column of Pia).
        #      And indicate cells that are a strong member of a CA (ie. pass a Threshold in Pia column).
        if ZorY=='z':
            e.set_alpha(0.5*Pia[i,A])
            if i < Nsplit: 
                e.set_facecolor('red')
            else:
                e.set_facecolor('blue') 
        #
        # (2). Plot all Cell RF's with ALPHA BASED ON FULL SPIKE RATE? Outline cells co-involved in CAs that a given Cell is a member of.
        #      with colors reflecting the CAs.
        elif ZorY=='y':             
            e.set_alpha(1)
            e.set_edgecolor('lightslategrey')
            e.set_facecolor('none')  
            e.set_linewidth(2)
            e.set_linestyle('-')
            ax.add_patch(e)
        else:
            'This should never happen.'

        if i < Nsplit:
            ax.text( STRF_gauss[i,1],STRF_gauss[i,3], i, ha='center', va='center', fontsize=10 )
        else:
            ax2.text( STRF_gauss[i,1],STRF_gauss[i,3], i, ha='center', va='center', fontsize=10 )


    # Plot a colorbar or alpha values matched up with Pia values, indicating our threshold for CA membership
    xmin = -5
    xmax = 45
    ymin = 5
    ymax = 40
    cbVal = np.linspace(0,1,5)
    cbVal = cbVal[1:]
    #
    CB = [ mpp.Ellipse( (xmin+i*0.7*(xmax-xmin), ymin+2), 3, 4, 90  ) for i in list( cbVal ) ] # parameters are: [Amplitude, x0, sigmax, y0, sigmay, angle(in rad)]
    for i,c in enumerate(CB):
        ax.add_artist(c)
        c.set_clip_box(ax.bbox)
        c.set_facecolor('red') 
        c.set_alpha(0.5*cbVal[i])
        c.set_linestyle('-')
        c.set_edgecolor('lightslategrey')
        c.set_linewidth(1) 
        ax.add_patch(c)  
        ax.text( xmin+cbVal[i]*0.7*(xmax-xmin), ymin+2, str(cbVal[i].round(2)).replace('0',''), ha='center', va='center', fontsize=10 )
    #
    if N > Nsplit:
        CB2 = [ mpp.Ellipse( (xmin+i*0.7*(xmax-xmin) + 0.1*(xmax-xmin) , ymin+2), 3, 4, 90  ) for i in list( cbVal ) ] # parameters are: [Amplitude, x0, sigmax, y0, sigmay, angle(in rad)]
        for i,d in enumerate(CB2):
            ax2.add_artist(d)
            d.set_clip_box(ax2.bbox)
            d.set_facecolor('blue') 
            d.set_alpha(0.5*cbVal[i])
            d.set_linestyle('-')
            d.set_edgecolor('lightslategrey')
            d.set_linewidth(1) 
            ax2.add_patch(d)  
            ax2.text( xmin + cbVal[i]*0.7*(xmax-xmin) + 0.1*(xmax-xmin), ymin+2, str(cbVal[i].round(2)).replace('0',''), ha='center', va='center', fontsize=10 )
    
    ax.text( xmin, ymin+2, str('$P_{ia}=$'), ha='left', va='center', fontsize=10 )   
    ax.text( xmax, ymin+2, str('$\Theta=$'+str(TH)), ha='right', va='center', fontsize=10 )   
    ax.plot([xmin,xmax],[ymin+4,ymin+4],'k--') 
    #
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks( np.linspace(xmin, xmax, 4) )
    ax.set_yticks( np.linspace(ymin, ymax, 4) )
    ax.invert_yaxis()
    ax.grid()
    #
    if N > Nsplit:
        ax2.text( xmin, ymin+2, str('$P_{ia}=$'), ha='left', va='center', fontsize=10 )   
        ax2.text( xmax, ymin+2, str('$\Theta=$'+str(TH)), ha='right', va='center', fontsize=10 )   
        ax2.plot([xmin,xmax],[ymin+4,ymin+4],'k--') 
        #
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)
        ax2.set_xticks( np.linspace(xmin, xmax, 4) )
        ax2.set_yticks( np.linspace(ymin, ymax, 4) )
        ax2.invert_yaxis()
        ax2.grid()


    return ax, ax2







def ellipse_RFz_multi_2ax(STRF_gauss, Pia, A, ax, ax2, Bs, cols, alf, Nsplit=None): 
# AsInB, ZorY, TH,


    N = STRF_gauss.shape[0]
    stretch=2
    ells = [ mpp.Ellipse( (STRF_gauss[i,1],STRF_gauss[i,3]), stretch*STRF_gauss[i,2], stretch*STRF_gauss[i,4], np.rad2deg(2*np.pi-STRF_gauss[i,5])  ) for i in range(N) ] # parameters in STRF_gauss are: [Amplitude, x0, sigmax, y0, sigmay, angle(in rad)]
    #

    if not Nsplit:
        Nsplit=N



    # If we are plotting cells belonging to CA, then plot edges fully dark with no face color. Because edges are depending on face alpha.
    #bb=0
    for i,e2 in enumerate(ells):
        if i <= Nsplit:
            ax.add_artist(e2)
            e2.set_clip_box(ax.bbox)
        else:
            #print(i, Nsplit)
            ax2.add_artist(e2)
            e2.set_clip_box(ax2.bbox)
        e2.set_facecolor('none') 
        e2.set_alpha(alf)
        e2.set_linestyle('-')
        #
        if i in Bs:
            #bb+=1     # to make ellipses slightly thinner so you can see ones underneath
            e2.set_edgecolor(cols)
            e2.set_linewidth(2) #*(1-0.1*bb) )
        else:
            ateam='go'
            # e2.set_edgecolor('lightslategrey')
            # e2.set_linewidth(1) 
        #    
        if i <= Nsplit:
            ax.add_patch(e2)
        else:
            ax2.add_patch(e2)


    for i,e in enumerate(ells):
        if i in Bs:
            if i <= Nsplit:
                ax.add_artist(e)
                e.set_clip_box(ax.bbox)
                ax.text( STRF_gauss[i,1]+0.5*np.random.random()*stretch*STRF_gauss[i,2], \
                    STRF_gauss[i,3]+0.5*np.random.random()*stretch*STRF_gauss[i,4], A, \
                    ha='center', va='center', fontsize=6, color=cols )
            else:
                ax2.add_artist(e)
                e.set_clip_box(ax2.bbox)
                ax2.text( STRF_gauss[i,1]+0.5*np.random.random()*stretch*STRF_gauss[i,2], \
                    STRF_gauss[i,3]+0.5*np.random.random()*stretch*STRF_gauss[i,4], A, \
                    ha='center', va='center', fontsize=6, color=cols )

            

    # Plot a colorbar or alpha values matched up with Pia values, indicating our threshold for CA membership
    xmin = -5
    xmax = 45
    ymin = 5
    ymax = 40
    cbVal = np.linspace(0,1,5)
    cbVal = cbVal[1:]
    #
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks( np.linspace(xmin, xmax, 4) )
    ax.set_yticks( np.linspace(ymin, ymax, 4) )
    ax.invert_yaxis()
    ax.grid()
    #
    if N > Nsplit:
        #  
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)
        ax2.set_xticks( np.linspace(xmin, xmax, 4) )
        ax2.set_yticks( np.linspace(ymin, ymax, 4) )
        ax2.invert_yaxis()
        ax2.grid()


    return ax, ax2






def ellipse_RFz_multiCelltype_1ax(STRF_gauss, Pia, A, ax, Bs, alf, Nsplit=None): 

    N = STRF_gauss.shape[0]
    stretch=2
    convertSTRF2mov = 1    # each monitor stim pixel is 15 RF pixels. THIS ISNT WORKING. FUCK IT FOR THESIS.
    ells = [ mpp.Ellipse( (convertSTRF2mov*STRF_gauss[i,1],convertSTRF2mov*STRF_gauss[i,3]), convertSTRF2mov*stretch*STRF_gauss[i,2], convertSTRF2mov*stretch*STRF_gauss[i,4], np.rad2deg(2*np.pi-STRF_gauss[i,5])  ) for i in range(N) ] # parameters in STRF_gauss are: [Amplitude, x0, sigmax, y0, sigmay, angle(in rad)]
    #



    if not Nsplit:
        Nsplit=N
  


    # If we are plotting cells belonging to CA, then plot edges fully dark with no face color. Because edges are depending on face alpha.
    for i,e2 in enumerate(ells):
        ax.add_artist(e2)
        e2.set_clip_box(ax.bbox)
        e2.set_facecolor('none') 
        e2.set_alpha(alf)
        e2.set_linestyle('-')
        #
        if i in Bs:
            if i <= Nsplit:
                e2.set_edgecolor('red')
                # ax.text( STRF_gauss[i,1]+0.5*np.random.random()*stretch*STRF_gauss[i,2], \
                #     STRF_gauss[i,3]+0.5*np.random.random()*stretch*STRF_gauss[i,4], A, \
                #     ha='center', va='center', fontsize=6, color='red' )
            else:
                e2.set_edgecolor('cyan')
                # ax.text( STRF_gauss[i,1]+0.5*np.random.random()*stretch*STRF_gauss[i,2], \
                #     STRF_gauss[i,3]+0.5*np.random.random()*stretch*STRF_gauss[i,4], A, \
                #     ha='center', va='center', fontsize=6, color='cyan' )
            e2.set_linewidth(2)
        else:
            ateam='go'
        #    
        ax.add_patch(e2)


    # for i,e in enumerate(ells):
    #     if i in Bs:
    #         if i <= Nsplit:
    #             ax.add_artist(e)
    #             e.set_clip_box(ax.bbox)
    #             ax.text( STRF_gauss[i,1]+0.5*np.random.random()*stretch*STRF_gauss[i,2], \
    #                 STRF_gauss[i,3]+0.5*np.random.random()*stretch*STRF_gauss[i,4], A, \
    #                 ha='center', va='center', fontsize=6, color='red' )
    #         else:
    #             ax2.add_artist(e)
    #             e.set_clip_box(ax2.bbox)
    #             ax2.text( STRF_gauss[i,1]+0.5*np.random.random()*stretch*STRF_gauss[i,2], \
    #                 STRF_gauss[i,3]+0.5*np.random.random()*stretch*STRF_gauss[i,4], A, \
    #                 ha='center', va='center', fontsize=6, color='cyan' )

            

    # Plot a colorbar or alpha values matched up with Pia values, indicating our threshold for CA membership
    xmin = -5*convertSTRF2mov
    xmax = 45
    ymin = 5
    ymax = 40
    cbVal = np.linspace(0,1,5)
    cbVal = cbVal[1:]
    #
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks( np.linspace(xmin, xmax, 4) )
    ax.set_yticks( np.linspace(ymin, ymax, 4) )
    ax.invert_yaxis()
    ax.grid()


    return ax






def plot_crispRobust_RFs_and_PSTH( numCAs, numIters, metricData, metricNames, psthZ_accum, binsPSTH, \
    Pia, RFs, Nsplit, cellsIn_collect, output_figs_dir, fname, figSaveFileType ):
    # TH, minMemb, maxMemb,

    N =Pia.shape[0]

    psthZ_sum = psthZ_accum.sum(axis=1)
    psthZ_normed = psthZ_accum.T/psthZ_sum[None,:]

    # set1 = plt.cm.Set1( np.arange(9) )
    # colsAll = set1

    #jetz = plt.cm.jet( np.arange(256) )
    #colsAll = jetz[ np.linspace(0,len(jetz)-1,numCAs).round().astype(int) ]

    srt_byMet = np.argsort(metricData[0]+metricData[1])[::-1] # Sort by sum of two metrics.

    # Build my own fucking colormap cause all these ones suck.
    colsAll = list()
    colsAll.append( [1,0,0,1] )
    colsAll.append( [0,1,0,1] )
    colsAll.append( [0,0,1,1] )
    colsAll.append( [0,1,1,1] )
    colsAll.append( [1,0,1,1] )
    colsAll.append( [0,0,0,1] )
    #colsAll.append( [1,1,0,1] ) # yellow sucks

    ordinals = ['$1^{st}$','$2^{nd}$','$3^{rd}$','$4^{th}$','$5^{th}$', \
                '$6^{th}$', '$7^{th}$', '$8^{th}$', '$9^{th}$', '$10^{th}$']


    for iternum in range(numIters):

        plt.rc('font', weight='bold', size=12)
        
        #
        # Plot RFs of Z in bottom right.
        if N > Nsplit:
            f = plt.figure(figsize=(30,5))
            axRF1   = plt.subplot2grid( (1,6), (0,0) )
            axRF2   = plt.subplot2grid( (1,6), (0,1) )
            axPS    = plt.subplot2grid( (1,6), (0,2), colspan=2 )
            axRC    = plt.subplot2grid( (1,6), (0,4) )
            axM2    = plt.subplot2grid( (1,6), (0,5) )
        else:
            f = plt.figure(figsize=(25,5))
            axRF1   = plt.subplot2grid( (1,5), (0,0) )
            axPS    = plt.subplot2grid( (1,5), (0,1), colspan=2 )
            axRC    = plt.subplot2grid( (1,5), (0,3) )
            axM2    = plt.subplot2grid( (1,5), (0,4) )
            axRF2   = None
            #
        
        #
        axRC.scatter(metricData[0], metricData[1], s=150, facecolors='none', edgecolors='black' )
        axM2.scatter(metricData[2], metricData[3], s=150, facecolors='none', edgecolors='black' )
        #
        for i,A in enumerate(srt_byMet[iternum*numCAs:(iternum+1)*numCAs]):
            # if N > Nsplit:
            axPS.plot( binsPSTH[:-1], psthZ_normed[:,A], color=colsAll[i], linewidth=3, label=str( str(A) + ' : ' + str( int(psthZ_sum[A]) ) + ' : ' + \
                str( metricData[0][A].round(2) ) + ' : ' +  str( metricData[1][A].round(2) ) + ' : ' +  str( metricData[2][A].round(2) )  + ' : ' +  str( metricData[3][A].round(2) ) ), alpha=0.5 )
            # else:
            #     axPS.plot( binsPSTH[:-1], psthZ_normed[:,A], color=colsAll[i], linewidth=3, label=str( str(A) + ' : ' + str( int(psthZ_sum[A]) ) + ' : ' + \
            #         str( metricData[0][A].round(2) ) + ' : ' +  str( metricData[1][A].round(2) ) ), alpha=0.5 )
            #
            axRC.scatter(metricData[0][A], metricData[1][A], s=100, color=colsAll[i], alpha=0.5)
            axM2.scatter(metricData[2][A], metricData[3][A], s=100, color=colsAll[i], alpha=0.5)
            #
            # PiaIndx = 1-Pia[:,A]
            # Bs = rc.find_good_ones(PiaIndx, TH, minMemb, maxMemb, ZorY='z')  
            # print(Bs)
            #
            Bs = cellsIn_collect[A]
            #
            axRF1,axRF2 = ellipse_RFz_multi_2ax(RFs, Pia, A, axRF1, axRF2, Bs, colsAll[i], 0.5, Nsplit)
        #
        axRC.errorbar( metricData[0].mean(), metricData[1].mean(), xerr= metricData[0].std(), yerr= metricData[1].std() )
        axRC.set_xlim(0,1)
        axRC.set_ylim(0,1)
        axRC.set_xlabel(metricNames[0])
        axRC.set_ylabel(metricNames[1])
        axRC.set_aspect('equal')
        axRC.grid(True)
        #
        axM2.errorbar( metricData[2].mean(), metricData[2].mean(), xerr= metricData[2].std(), yerr= metricData[2].std() )
        axM2.set_xlim(0,1)
        axM2.set_ylim(0,1)
        axM2.set_xlabel(metricNames[2])
        axM2.set_ylabel(metricNames[3])
        axM2.set_aspect('equal')
        axM2.grid(True)
        #
        #if N > Nsplit:
        axPS.legend(loc='best', title=str('z  :  #Act  :  ' + metricNames[0] + '  :  ' +  metricNames[1] + '  :  ' +  metricNames[2] + '  :  ' +  metricNames[3] ) ) #, bbox_to_anchor=(1, 0.5))
        #else:
        #    axPS.legend(loc='best', title=str('z  :  #Act  :  ' + metricNames[0] + '  :  ' +  metricNames[1]) )
        axPS.set_xlim(0,binsPSTH.max())
        #axPS.set_ylim(0,1)
        axPS.set_xlabel('Time (ms)')
        axPS.set_ylabel('PSTH (normed)')
        axPS.set_aspect('auto')
        axPS.grid(True)
        #
        axRF1.set_xticklabels([])
        axRF1.set_yticklabels([])
        axRF1.set_aspect('equal')
        axRF1.grid(True)
        axRF1.set_xlabel( str('RFs (ct1)'), fontsize=16 )
        if N > Nsplit:
            axRF2.set_xticklabels([])
            axRF2.set_yticklabels([])
            axRF2.set_xlabel( str('RFs (ct2)'), fontsize=16 )
            axRF2.set_aspect('equal')
            axRF2.grid(True)

        plt.suptitle( str( r'order' + str(iternum+1) + ' ' + str(numCAs) + 'CAs : ' + fname.replace('_',' ') ) )
        #
        # SAVE PLOT.                                                    # Info for saving the plot.
        CrispRobustCAs_figs_save_dir = str( output_figs_dir+'../CA_RF_PSTHs_compareMetrics'+metricNames[0]+'v'+metricNames[1]+'/')
        #
        if not os.path.exists( str(CrispRobustCAs_figs_save_dir) ):
            os.makedirs( str(CrispRobustCAs_figs_save_dir) )

        plt.savefig( str(CrispRobustCAs_figs_save_dir + fname + '_iter' + str(iternum) + '.' + figSaveFileType ) ) 
        #plt.show()
        plt.close() 

    return 










   
def psth_YorZ_andItsMembers(mainV, Pia, TH, minMemb, maxMemb, psthY, psthZ, smoothKern, cols, PSTH_figs_dir, model_file, figSaveFileType):
    
    if mainV=='y':
        subV='z'
        subType = 'CA'
        mainType = 'Cell'
        psthA = psthY
        psthB = psthZ
        #cols = colsBright
        mrkMain = ['.',5, 0.9,'dimgrey']
        mrkSub = ['o', 10, 0.3]
    elif mainV=='z':
        subV='y'
        subType = 'Cell'
        mainType = 'CA'
        psthA = psthZ
        psthB = psthY
        #cols=colsDark
        mrkMain = ['o', 10, 0.9, 'red']
        mrkSub = ['.',5, 0.1]
    else:
        print('I dont understand variable type. Should be ''y'' for cell or ''z'' for CA')
        return

    minT = 0                # max and min times on PSTH plots.
    maxT = psthA.shape[2]     
    alf = 0.8

    As_multi = rc.find_good_ones(Pia, TH, minMemb, maxMemb, ZorY=mainV) # main variable. If main=Y, then plot PSTH of Cell and all CA's it participates in (ie. sub=Z). 
                                                                        # sub variable.  If main=Z, then plot PSTH of CA and all Cells that participate in it (ie. sub=Y)   

    PSTH_Gauss_smoothing = sig.windows.general_gaussian(smoothKern,1,1)

    for A in As_multi:  
        #
        plt.rc('font', weight='bold', size=18)
        f,ax = plt.subplots(3, 1, figsize=(20,10))
        #
        # Plot PSTH of Main Variable. Either Cell Y or CA Z.
        if mainV=='y':
            PiaIndx = Pia[A]
        elif mainV=='z':
            PiaIndx = Pia[:,A]
        else:
            'This should never happen.'
        Bs = rc.find_good_ones(PiaIndx, TH, minMemb, maxMemb, ZorY=subV)
        infrd = np.zeros_like(Bs)
        for i, B in enumerate(Bs):
            infrd[i] = psthB[B].sum().astype(int)
        infrdSort = np.argsort(infrd)[::-1] 
        Bs = Bs[infrdSort]
        #
        Asmooth = np.convolve( psthA[A].sum(axis=0), PSTH_Gauss_smoothing )
        AsmoothN = Asmooth/Asmooth.sum()
        ax[0].plot( AsmoothN, color=mrkMain[3], linestyle='-', linewidth=2, label=str(mainV+'#'+str(A)+ \
            ' -> '+subV+'='+str(Bs) +' (#='+str(psthA[A].sum().astype(int))+')') ) #' with Pi='+str(np.round(Pi[y],2))+ \
        ax[0].text( maxT, AsmoothN.max(), str(mainV+'='+str(A)+' (#='+str(psthA[A].sum().astype(int))+')'), \
                                color=mrkMain[3], horizontalalignment='right', verticalalignment='top' )
        #
        # Plot PSTH of all Cell Assemblies Z that Y participates in.
        done_Bs = list()
        onset_Bs = list()
        jj=0             # done_Bs and jj are to plot difference between PSTHs subVs
        for i,B in enumerate(Bs):
            done_Bs.append(B)
            if subV=='y':
                PiaIndx = Pia[B]
            elif subV=='z':
                PiaIndx = Pia[:,B]
            else:
                'This should never happen.'
            #
            As = rc.find_good_ones(PiaIndx, TH, minMemb, maxMemb, ZorY=subV)
            if As.size==0:
                As=np.array(A)
            #else:
                #if mainV=='y':
            Bsmooth = np.convolve( psthB[B].sum(axis=0), PSTH_Gauss_smoothing )
            BsmoothN = Bsmooth/Bsmooth.sum()
            #
            if BsmoothN[500:].sum()/BsmoothN.sum() > 0.2:  
                ax[1].plot( BsmoothN, color=cols[i], linestyle='--', linewidth=2, alpha = alf, label=str(subV+'#'+str(B)+ \
                    ' -> '+mainV+'='+str(As) +' (#='+str(psthB[B].sum().astype(int))+')') )
            else:
                onset_Bs.append(B)

            # WORKING HERE :: PLOT COLORED TEXT INSTEAD OF LEGEND. LIKE RASTER PLOTS.
            ax[0].text( maxT, AsmoothN.max()*(0.85-0.15*i), str(subV+'#'+str(B)+' -> '+mainV+'='+str(As) +' (#='+str(psthB[B].sum().astype(int))+')'), \
                                color=cols[i], horizontalalignment='right', verticalalignment='top' )          
            

            #
            # Plot difference between pairs of cell assembly PSTHs
            for B2 in Bs[Bs!=B]:
                if not (B2 in done_Bs):
                    #
                    if subV=='y':
                        PiaIndx1 = Pia[B2]
                        PiaIndx2 = Pia[B]
                    elif subV=='z':
                        PiaIndx1 = Pia[:,B2]
                        PiaIndx2 = Pia[:,B]
                    else:
                        'This should never happen.'
                    As1 = rc.find_good_ones(PiaIndx1, TH, minMemb, maxMemb, ZorY=subV)
                    As2 = rc.find_good_ones(PiaIndx2, TH, minMemb, maxMemb, ZorY=subV)
                    if As1.size==0:
                        As1=np.array(A)
                    if As2.size==0:  
                        As2=np.array(A)
                    jj+=1
                    B2smooth = np.convolve( psthB[B2].sum(axis=0), PSTH_Gauss_smoothing )
                    B2smoothN = B2smooth/B2smooth.sum()
                    #
                    if not (B2smoothN[500:].sum()/B2smoothN.sum() > 0.2):  
                        onset_Bs.append(B2)


                    if not( (B in onset_Bs) or (B2 in onset_Bs) ):
                        ax[2].plot( np.abs(BsmoothN-B2smoothN), color=cols[jj], linestyle='-', linewidth=2, alpha = alf, \
                            label=str('|'+subV+str(B)+' - '+subV+str(B2)+'|' ) )

                


        plt.suptitle( str('PSTH: '+mainV+'#'+str(A)+' and '+subV+'s='+str(Bs) ) )
        xtks = np.arange(minT,maxT+1,500)
        ax[0].grid()
        ax[0].set_ylabel( str(mainType + ' ' + mainV ) )
        #ax[0].legend(fontsize=12)           
        ax[0].set_xticks( xtks)
        ax[0].set_xticklabels( np.round(xtks/1000, 2), fontsize=10 )    
        ax[0].set_xlim(minT,maxT)
        #
        ax[1].grid()
        ax[1].set_ylabel(  str(subType + ' ' + subV )  )
        #ax[1].legend(fontsize=12)
        ax[1].set_xticks( xtks )
        ax[1].set_xticklabels( np.round(xtks/1000, 2), fontsize=10 )    
        ax[1].set_xlim(minT,maxT)
        #
        ax[2].grid()
        ax[2].set_ylabel( str('diff '+subType+'s $\Delta$'+subV) )
        ax[2].legend(fontsize=16)  
        ax[2].set_xticks( xtks )
        ax[2].set_xticklabels( np.round(xtks/1000, 2), fontsize=10 )    
        ax[2].set_xlim(minT,maxT)
        ax[2].set_xlabel( str('Time (sec)') )

        plt.tight_layout()
        plt.savefig( str(PSTH_figs_dir + 'PSTH_'+model_file.replace('LearnedModel_','').replace('.npz','')+'_'+mainV+str(A)+'_'+subV+'s'+str(Bs)+'.'+figSaveFileType ) )
        plt.close()     

    return




def raster_YorZ_andItsMembers(mainV, Pia, TH, minMemb, maxMemb, psthY, psthZ, STRF_gauss, mn_CosSim_accModels, pyGLM_cs, pyGLM_binning, \
                    Zmatch_gather, Pia_gather, PSTHz_gather, binsPSTH, cols, PSTH_figs_dir, model_file, Nsplit, Robust, CrispX, figSaveFileType):
                    # CA_coactivity_allSWs, Cell_coactivity_allSWs,
    
    if mainV=='y':
        subV='z'
        sz = Pia.shape[0] #N
        subType = 'CA'
        mainType = 'Cell'
        #CoAct = Cell_coactivity_allSWs
        psthA = psthY
        psthB = psthZ
        mrkMain = ['.',5, 0.9,'dimgrey']
        mrkSub = ['o', 10, 0.3]
    elif mainV=='z':
        subV='y'
        sz = Pia.shape[1] #M
        subType = 'Cell'
        mainType = 'CA'
        #CoAct = CA_coactivity_allSWs
        psthA = psthZ
        psthB = psthY
        mrkMain = ['o', 10, 0.9, 'red']
        mrkSub = ['.',5, 0.1]
    else:
        print('I dont understand variable type. Should be ''y'' for cell or ''z'' for CA')
        #return


    numTrials = len(psthA) #psthA.shape[1]  
    #print('NumTrials is NOT ',numTrials,'. in raster_YorZ_andItsMembers. Fix it.')  



    # number of times each CA is active
    zRate = np.zeros( len(PSTHz_gather) )
    for i in range(len(PSTHz_gather)):
        zRate[i] = PSTHz_gather[i][:,0].sum() 
    #zRate = int(zRate) 


    # Compute average temporal cosine similarity between za activations in this model and other best matching zs in other models. 
    rasFiles = np.where( np.any(PSTHz_gather[0],axis=0) )[0] # if rasterfile wasnt there, values will be all zeros
    mnCosSimTemp = np.zeros( len(PSTHz_gather) ) 
    for i in range(len(PSTHz_gather)):
        csT = cosine_similarity(PSTHz_gather[i].T,PSTHz_gather[i].T) 
        mnCosSimTemp[i] = csT[0,rasFiles[1:]].mean() # [1:] so you dont grab cosSim with itself. CA itself is in 0 position.



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    #As = rc.find_good_ones(Pia, TH, minMemb, maxMemb, ZorY=mainV) # main variable. If main=Y, then plot PSTH of Cell and all CA's it participates in (ie. sub=Z). 
    #print('As ',As)                                               # sub variable.  If main=Z, then plot PSTH of CA and all Cells that participate in it (ie. sub=Y)   
    #                                                             # Comments below arefor mainV='z'. Meaning A pertains to CAs, B pertains to Cells
    #                                                             #      and AsInB are the other CAs that cells in this CA also participate in.
    #
    As = np.arange(sz) # Do this for every CA without testing for goodness at this stage.

    infrdA = np.zeros_like(As)
    for i, A in enumerate(As):
        infrdA[i] = np.array([len(psthA[tr][A]) for tr in range(numTrials)]).sum().astype(int)
    infrdASort = np.argsort(infrdA)[::-1] 


    #
    for ii, A in enumerate(As):                                                 # Make a seperate figure for each CA that is                                 
        if mainV=='y':
            PiaIndx = Pia[A]
        elif mainV=='z':
            PiaIndx = Pia[:,A]
        else:
            'This should never happen.'
        Bs = rc.find_good_ones(PiaIndx, TH, minMemb, maxMemb, ZorY=subV)        # Find cells within a CA that pass a threshold membership participation probability.
        #print('Bs ',Bs)
        infrdB = np.zeros_like(Bs)
        for i, B in enumerate(Bs):
            infrdB[i] = np.array([len(psthB[tr][B]) for tr in range(numTrials)]).sum().astype(int)
        infrdBSort = np.argsort(infrdB)[::-1] 
        #Bs = Bs[infrdSort]





        #
        plt.rc('font', weight='bold', size=12)
        f = plt.figure(figsize=(20,10))
        plt.suptitle( str( 'Model' + Zmatch_gather[0][0][0] + ' ' + mainV + str(A)+' --> Active '+str(infrdA[ii])+'x' + ', $<cs_X>$=' + str(mn_CosSim_accModels[A].round(2)) + ', $<cs_T>$=' + str(mnCosSimTemp[A].round(2)) + ', $\Delta cs_{GLM}$=' + str(pyGLM_cs[A].round(2) )), fontsize=20 )
        #
        # Raster of cell assembly, Z.
        for k in range(2):
            minT = 2500*k
            maxT = 2500*(k+1)
            axx = plt.subplot2grid( (5,1), (k,0) ) #, rowspan=2 )
            # if k>=3:
            #     axx = plt.subplot2grid( (5,10), (k,0), colspan=3 )
            #     minT = 4500 + 500*(k-3)
            #     maxT = 4500 + 500*(k-2)



            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            #
            # Raster of cells Y involved in cell assembly, Z.
            AsInB = list()
            for i, B in enumerate(Bs):
                for tr in range(numTrials):
                    ind = np.where( np.bitwise_and( np.array(psthB[tr][B])>minT, np.array(psthB[tr][B])<maxT ) )[0]

                    if len(ind)>1:
                        axx.scatter(psthB[tr][B][ind[0]:ind[-1]], tr*np.ones_like(psthB[tr][B][ind[0]:ind[-1]]), \
                            s=mrkSub[1], marker=mrkSub[0], color=cols[i], alpha=mrkSub[2], \
                            label=str(subV+'='+str(B) )) # +' (#='+str(psthB[B].sum().astype(int))+')')) 

                if k==0:
                    #
                    if subV=='y':
                        PiaIndx = Pia[B]
                    elif subV=='z':
                        PiaIndx = Pia[:,B]
                    else:
                        'This should never happen.'
                    #   
                    xx = rc.find_good_ones(PiaIndx, TH, minMemb, maxMemb, ZorY=subV)
                    AsInB.append(xx)
                    if AsInB[i].size==0:
                        AsInB[i]=np.array([A])
                    axx.text( maxT, 40+30*i, str(subV+'='+str(B)+' -> '+mainV+'='+str(AsInB[i]) ), color=cols[i], ha='right', va='top' ) # +' (#='+str(psthB[B].sum().astype(int))+')'
                    #           
                    axx.text( maxT, 10, str(mainV+'='+str(A)+' -> '+subV+'='+str(Bs) ), color=mrkMain[3], ha='right', va='top' ) # +' (#='+str(psthA[A].sum().astype(int))+')'
        



            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            #
            # Raster of cell assembly, Z.
            for tr in range(numTrials):
                ind = np.where( np.bitwise_and( np.array(psthA[tr][A])>minT, np.array(psthA[tr][A])<maxT ) )[0]
                # try:
                #print('A ind', ind)
                if len(ind)>1:
                    axx.scatter(psthA[tr][A][ind[0]:ind[-1]], tr*np.ones_like(psthA[tr][A][ind[0]:ind[-1]]), \
                        s=mrkMain[1], marker=mrkMain[0], color=mrkMain[3], alpha=mrkMain[2], linewidth=0.5, \
                        facecolor=None, label=str(mainV+'='+str(A) ) ) # +' (#='+str(psthA[A].sum().astype(int))+')'))
                # except:
                #     print('ok ',ind)
            
            #axx.scatter(t[ind],tr[ind], s=mrkMain[1], marker=mrkMain[0], color=mrkMain[3], alpha=mrkMain[2], linewidth=0.5, facecolor=None, label=str(mainV+'='+str(A)+' (#='+str(psthA[A].sum().astype(int))+')'))
            axx.set_xticks( np.arange(minT,maxT+1,500))
            axx.set_xticklabels( np.round(np.arange(minT,maxT+1,500)/1000, 2), fontsize=10 )    
            axx.set_xlim(minT,maxT)
            axx.set_ylim(0,numTrials)
            axx.invert_yaxis()
            axx.grid()









        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        #
        # Plot temporal profiles of activations of CAs in other models that are 
        # matched to this one spatial cosine similarity
        #
        PSTHz_sum = PSTHz_gather[A].sum(axis=0)
        PSTHz_normed = PSTHz_gather[A].T/PSTHz_sum[:,None]

        axT = plt.subplot2grid( (5,1), (2,0) )
        axT.imshow(PSTHz_normed[rasFiles])

        xt = np.arange( 0, len(binsPSTH)-1, 10 )
        axT.set_xticks( xt )
        axT.set_xticklabels( (binsPSTH[xt]/1000).round(1), fontsize=10 ) # , rotation=45
        axT.set_xlabel('time (sec)')
        for i in range( len(rasFiles) ):
            axT.text( xt.max()+1, i, str( int(PSTHz_sum[i]) ), ha='left', color='white')
        axT.set_yticks( np.arange( len(rasFiles) ) )
        axT.set_yticklabels( Zmatch_gather[A], fontsize=10 )
        axT.set_ylabel('$z_a$ activations')# , fontsize=14, fontweight='bold')
        #ax.set_ylabel('model / CA')
        axT.set_title( str( np.nanmax(PSTHz_normed).round(2) ) )
        axT.set_aspect('auto')
        #


        # TRYING TO ADD A 2ND AXIS TO PUT RFS FOR CELLTYPE 2 CELLS.
        #
        # Plot RFs of Z in bottom right.
        N = Pia.shape[0]
        if N > Nsplit:
            axCO    = plt.subplot2grid( (5,9), (3,0), rowspan=2, colspan=2 )
            axC2    = plt.subplot2grid( (5,9), (3,2), rowspan=2, colspan=2 )
            axRF1   = plt.subplot2grid( (5,9), (3,4), rowspan=2, colspan=2 )
            axRF2   = plt.subplot2grid( (5,9), (3,6), rowspan=2, colspan=2 )
            axRF2.set_aspect('equal')
            axG     = plt.subplot2grid( (5,9), (3,8), rowspan=2)
            
        else:
            axCO    = plt.subplot2grid( (5,7), (3,0), rowspan=2, colspan=2 )
            axC2    = plt.subplot2grid( (5,7), (3,2), rowspan=2, colspan=2 )
            axRF1   = plt.subplot2grid( (5,7), (3,4), rowspan=2, colspan=2 )
            axRF2   = None 
            axG     = plt.subplot2grid( (5,7), (3,6), rowspan=2 )
        #
        axRF1.set_aspect('equal')
        #
        print('A = ',A)
        print('Bs = ',Bs)
        print('AsInB = ',AsInB)
        #
        axRF1,axRF2 = ellipse_RFz_2ax( STRF_gauss, Pia, A, axRF1, axRF2, Bs, AsInB, cols, TH, ZorY=mainV, Nsplit=Nsplit )
        axRF1.set_xticklabels([])
        axRF1.set_yticklabels([])
        axRF1.set_xlabel( str('RFs'), fontsize=16 )
        if N > Nsplit:
            axRF2.set_xticklabels([])
            axRF2.set_yticklabels([])
            axRF2.set_xlabel( str('RFs (ct2)'), fontsize=16 )



        # #
        # # Scatter plot zRate vs. dp(y). 
        # axCO.scatter(zRate, pyGLM_cs, 50, alpha=0.5)  
        # for jj in range(sz):
        #     axCO.text( zRate[jj], pyGLM_cs[jj], str(jj), fontsize=12 )
        # axCO.text( zRate[A], pyGLM_cs[A], str(A), color='red', fontsize=20 )
        # axCO.set_xlabel('$z_a$ activations')# , fontsize=14, fontweight='bold')
        # axCO.set_ylabel( str(r'$\Delta p(y)_{null}$'))# , fontsize=14, fontweight='bold' )
        # axCO.grid()


        #
        # Scatter dp(y) how dp(y) changes with binning
        axCO.scatter(pyGLM_cs, pyGLM_binning, 50, alpha=0.5)  
        axCO.plot([0,1],[0,1],'k--')
        for jj in range(sz):
            axCO.text( pyGLM_cs[jj], pyGLM_binning[jj], str(jj), fontsize=10 )
        axCO.text(pyGLM_cs[A], pyGLM_binning[A], str(A), color='red', fontsize=20,  ha='center', va='center' )
        axCO.set_xlabel( str(r'$\Delta p(y)_{null}$ 1ms'))# , fontsize=14, fontweight='bold' )
        axCO.set_ylabel( str(r'$\Delta p(y)_{null}$ 100ms'))# , fontsize=14, fontweight='bold' )
        axCO.grid()
        axCO.set_ylim(0,1)
        axCO.set_xlim(0,1)
        axCO.set_aspect('equal')


        #
        # Scatter plot Robustness and Spatial Crispness Metrics (location defined above depending on 1 cell type or 2.)
        axC2.scatter(CrispX, Robust, 25, alpha=0.3)  
        for jj in range(sz):
            axC2.text( CrispX[jj], Robust[jj], str(jj), fontsize=10 )
        axC2.text( CrispX[A], Robust[A], str(A), color='red', fontsize=20, ha='center', va='center' )
        axC2.set_ylim([0,1])
        axC2.set_xlim(0,1) #np.max([1,CrispX.max()]))
        axC2.set_aspect('equal')
        axC2.set_ylabel('Robustness across models.')#, fontsize=14, fontweight='bold')
        axC2.set_xlabel('Spatial Crispness')#, fontsize=14, fontweight='bold' )
        axC2.grid()







        #
        # Plot other cell assemblies in other models that match this one.
        # Pia_gather # M x N x nMods. List of NxnMods Array. One for each CA # Cell assemblies that match in other models 
        # Zmatch_gather # list of strings M x nMods. Contains other model and other CA this one is matched to.
        axG.imshow(Pia_gather[A], vmin=0, vmax=1)
        axG.plot([-0.5,Pia_gather[A].shape[1]],[Nsplit,Nsplit],'w--')
        axG.set_xlim(-0.5,Pia_gather[A].shape[1]-1+0.5) # numModels
        axG.set_ylim(-0.5,Pia_gather[A].shape[0]+0.5)   # N - num cells
        axG.set_aspect('auto')
        axG.set_ylabel( 'cell ID $y_i$') #, fontsize=14, fontweight='bold' )
        axG.set_xlabel( 'matching CAs in other models') #, fontsize=14, fontweight='bold' )
        axG.set_yticks( [0,Pia_gather[A].shape[0]] )
        axG.set_xticks( np.arange( len(Zmatch_gather[A]) ) )
        axG.set_xticklabels( Zmatch_gather[A], fontsize=10, rotation=45 )
        #axG.title( str('M#'+str(aaa)+', z#'+str(iii)+', <cs> = '+str(mean_CS_accModels[aaa,iii].round(2))) )



        #plt.tight_layout()
        plt.savefig( str(PSTH_figs_dir +'SpkRaster_'+model_file.replace('LearnedModel_','').replace('.npz','')+'_'+mainV+str(A)+'_'+subV+'s'+str(Bs)+'.'+figSaveFileType ) ) 
        plt.close() 

    return



def human_readable(input):
    # Making axis labels and numbers human readable. Meh, Not worth it right now.

    output = list()

    for i in range(2):
        find_0 = (input/1e3**i      >= 1) # things less than 1000.
        find_K = (input/1e3**(i+1)  >= 1) # things to tag with Kilo
        ind = list( np.where( np.bitwise_and(find_0, np.bitwise_not(find_K) ) )[0] )
        output.append( [ input[x]/1e3**i for x in ind] )


    tags=['','K','M','G','T','P','FuckTon']

    return output, tags





def plot_xValidation(pJoints, pj_labels, plt_save_dir, fname_EMlrn, figSaveFileType, kerns=[1000,5000]):
    # Plot and save a figure with pjoint curves smoothed with some gaussian kernels for cross validation

    plt.figure( figsize=(20,10) ) # size units in inches
    plt.rc('font', weight='bold', size=16)
    #
    inds = np.arange( pJoints.shape[1] )
    #
    A = 0 # alpha value for plots where it will grow more opaque with smoother kernels.
    for kern in kerns:
        A+=0.3
        for ii in range(pJoints.shape[0]):
            smoo = sig.windows.general_gaussian(kern,1,kern/6)
            plt.plot(inds[:np.int(-1-kern)], np.convolve( pJoints[ii], smoo )[np.int(kern):np.int(-kern)]/smoo.sum(), alpha=A, linewidth=3, label=str( pj_labels[ii]+', $\sigma$='+str(kern) ) )
    #
    plt.legend()
    plt.grid()
    plt.xlabel('EM iteration #')
    plt.ylabel('A.U. - log probability (joint or conditional)')
    plt.title( str('p(Y,Z) when Z inferred smoothed w/ Gaussian Kernel smoothing') )
    plt.tight_layout()
    #
    if not os.path.exists( str(plt_save_dir + 'CrossValidation/') ):
        os.makedirs( str(plt_save_dir + 'CrossValidation/') )
    plt.savefig( str(plt_save_dir + 'CrossValidation/' + fname_EMlrn + '.' + figSaveFileType) )
    plt.close() 






# def plot_xValidationOLD(Z_inferred_train, pjoint_train, pjoint_test, plt_save_dir, fname_EMlrn, kerns=[1000,5000]):
#     # Plot and save a figure with pjoint curves smoothed with some gaussian kernels for cross validation

#     print( 'Plotting Cross validation.' )
#     t0 = time.time()
#     #
#     # numZinfrd = np.array([len(Z_inferred_train[xx]) for xx in range(len(Z_inferred_train))])
#     # inds = np.where(numZinfrd>0)[0]                 # <<<<------------ WHAT IF I DONT DO THIS?
#     inds = np.arange( len(pjoint_train) )
#     # kerns = [1000, 5000] #, 100, 300, 600, 1000]
#     plt.figure( figsize=(20,10) ) # size units in inches
#     plt.rc('font', weight='bold', size=16)
#     for kern in kerns:
#         smoo = sig.windows.general_gaussian(kern,1,kern/6)
#         #
#         try:
#             # plt.plot(inds[:-1], np.convolve( pjoint_train[inds], smoo )/smoo.sum(), alpha=0.5, label='train')
#             # plt.plot(inds[:-1], np.convolve( pjoint_test[inds], smoo )/smoo.sum(), alpha=0.5, label='test')
#             #
#             plt.plot(inds[:np.int(-1-kern)], np.convolve( pjoint_train[inds], smoo )[np.int(kern):np.int(-kern)]/smoo.sum(), alpha=0.8, linewidth=2, label=str('train, $\sigma$='+str(kern) ) )
#             plt.plot(inds[:np.int(-1-kern)], np.convolve( pjoint_test[inds], smoo )[np.int(kern):np.int(-kern)]/smoo.sum(), alpha=0.8, linewidth=2, label=str('test, $\sigma$='+str(kern) ) )
#         except:
#             plt.plot(inds[:np.int(-1-kern)], np.convolve( pjoint_train[inds], smoo )[np.int(kern):np.int(-kern)]/smoo.sum(), alpha=0.8, linewidth=2, label=str('train, $\sigma$='+str(kern) ) )
#         #
#     plt.legend()
#     plt.grid()
#     plt.xlabel('EM iteration #')
#     plt.ylabel('A.U. - pjoint kinda')
#     plt.title( str('p(Y,Z) when Z inferred smoothed w/ Gaussian Kernel smoothing') )
#     #plt.show()
#     #
#     if not os.path.exists( str(plt_save_dir + 'CrossValidation/') ):
#         os.makedirs( str(plt_save_dir + 'CrossValidation/') )
#     plt.savefig( str(plt_save_dir + 'CrossValidation/' + fname_EMlrn + '.pdf') )
#     plt.close() 
#     #
#     t1 = time.time()
#     print('Done w/ Cross validation. : time = ',t1-t0)



def plot_pairwise_model_CA_match(modelPair_cosSim, modelPair_cosSimNM, modelAndGT_cosSim, modelAndGT_cosSimNM, meanLogCond, GTtag, fnames, fparams, plt_save_dir, fname_save, measureTitle, figSaveFileType):

    vmaxx = 1#0.2#5 #np.max( [ modelPair_cosSim.max(), modelAndGT_cosSim.max(), modelPair_lenDif.max() ] )
    #
    if fnames[0].find('EM_model_data_')>=0:
    # Synth Data
        flabels = [ fp.replace('rand','Mod') for fp in fparams ] 
        ftit = fnames[0][(fnames[0].find('EM_model_data_')+14):fnames[0].find('_rand')].replace('_',' ').replace('pt','.')
    else:
    # Real Data                                   
        flabels = [ fp.replace('rand','Mod') for fp in fparams ]
        ftit = fnames[0][(fnames[0].find('LearnedModel_')+13):fnames[0].find('_rand')].replace('_',' ').replace('pt','.')
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    #
    f = plt.figure( figsize=(20,10) ) # size units in inches
    plt.rc('font', weight='bold', size=22)
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']
    if modelPair_cosSim.shape[0]<=6:
        numsFontSize = 24
    elif modelPair_cosSim.shape[0]<=12:
        numsFontSize = 18
    else:
        numsFontSize = 12
    #

    ax1 = plt.subplot2grid((1,1),(0,0))#,rowspan=5,colspan=5) 
    im = ax1.imshow(modelPair_cosSim-modelPair_cosSimNM, vmin=0, vmax=vmaxx, cmap='jet')
    ax1.set_xticks(np.arange(0, len(fnames)))
    ax1.set_xticklabels(flabels, fontsize=numsFontSize, fontweight='bold', rotation='45')
    ax1.set_yticks(np.arange(0, len(fnames)))
    ax1.set_yticklabels(flabels, fontsize=numsFontSize, fontweight='bold')
    # ax1.set_xlabel('model A', fontsize=22, fontweight='bold')
    # ax1.set_ylabel('model B', fontsize=22, fontweight='bold')
    ax1.set_title( 'Cosine Similarity (Matched - NM)', fontsize=22, fontweight='bold', ha="center", va="bottom" )
    # Loop over data dimensions and create text annotations.
    for i in range(len(fnames)):
        for j in range(len(fnames)):
            if i!=j:

                text = ax1.text(j,      i,      str((modelPair_cosSim[i,j]-modelPair_cosSimNM[i,j]).round(2)).replace('0.','.'), \
                                            ha="center", va="center", color="k", fontsize=numsFontSize, fontweight='bold' )
                text = ax1.text(j+0.02, i+0.02, str((modelPair_cosSim[i,j]-modelPair_cosSimNM[i,j]).round(2)).replace('0.','.'), \
                            ha="center", va="center", color="w", fontsize=numsFontSize, fontweight='bold' )
                text = ax1.text(j+0.04, i+0.04, str((modelPair_cosSim[i,j]-modelPair_cosSimNM[i,j]).round(2)).replace('0.','.'), \
                            ha="center", va="center", color="k", fontsize=numsFontSize, fontweight='bold' )

            else:

                meanish = (modelPair_cosSim[i]-modelPair_cosSimNM[i]).sum()/ (len(fnames) - 1)
                text = ax1.text(j,      i,      str(meanish.round(2)).replace('0.','.'), \
                            ha="center", va="center", color="k", fontsize=numsFontSize, fontweight='bold' )
                text = ax1.text(j+0.02, i+0.02, str(meanish.round(2)).replace('0.','.'), \
                            ha="center", va="center", color="w", fontsize=numsFontSize, fontweight='bold' )
                text = ax1.text(j+0.04, i+0.04, str(meanish.round(2)).replace('0.','.'), \
                            ha="center", va="center", color="k", fontsize=numsFontSize, fontweight='bold' )
                #
                # Mean Log Conditional probability of all spike words.
                text = ax1.text(-2, i, str(meanLogCond[i][0].round(4)), \
                            ha="center", va="center", color="k", fontsize=numsFontSize, fontweight='bold' )

    text = ax1.text(-2, -0.5, str(r'$<p(y \vert z)>_{data}$'), \
                            ha="center", va="bottom", color="k", fontsize=numsFontSize, fontweight='bold' )            



            

    #
    # #
    #
    ax2 = f.add_axes([0.72, 0.11, 0.1, 0.77])
    ax2.imshow(modelAndGT_cosSim[:,:1]-modelAndGT_cosSimNM[:,:1], vmin=0, vmax=vmaxx, cmap='jet' )
    #ax2.set_title('w/ GT', fontsize=22, fontweight='bold')#, ha="right", va="bottom" )
    ax2.set_xticks([0])
    ax2.set_xticklabels([GTtag], fontsize=numsFontSize, fontweight='bold', rotation=45)
    ax2.set_yticks([]) #np.arange(0, len(fnames)))
    #ax2.set_yticklabels(np.arange(0, len(fnames)), fontsize=20, fontweight='bold')
    # Loop over data dimensions and create text annotations.
    for i in range(len(fnames)):
        text = ax2.text(0,      i,      str((modelAndGT_cosSim[i,0]-modelAndGT_cosSimNM[i,0]).round(2)).replace('0.','.'), \
                                    ha="center", va="center", color="k", fontsize=numsFontSize,fontweight='bold' )
        text = ax2.text(0+0.04, i+0.04, str((modelAndGT_cosSim[i,0]-modelAndGT_cosSimNM[i,0]).round(2)).replace('0.','.'), \
                                    ha="center", va="center", color="k", fontsize=numsFontSize, fontweight='bold' )
        text = ax2.text(0+0.02, i+0.02, str((modelAndGT_cosSim[i,0]-modelAndGT_cosSimNM[i,0]).round(2)).replace('0.','.'), \
                                    ha="center", va="center", color="w", fontsize=numsFontSize, fontweight='bold' )
    # #
    # # #
    # #
    # ax4 = f.add_axes([0.58, 0.16, 0.05, 0.66])
    # ax4.imshow(modelAndGT_cosSimNM[:,:1], vmin=0, vmax=vmaxx, cmap='jet' )
    # ax4.set_title('NM w/ GT', fontsize=18, fontweight='bold')#, ha="right", va="bottom" )
    # ax4.set_xticks([])
    # ax4.set_yticks(np.arange(0, len(fnames)))
    # ax4.set_yticklabels(np.arange(0, len(fnames)), fontsize=20, fontweight='bold')
    # # Loop over data dimensions and create text annotations.
    # for i in range(len(fnames)):
    #     text = ax4.text(0,      i,      str(modelAndGT_cosSimNM[i,0].round(2)).replace('0.','.'), \
    #                                 ha="center", va="center", color="k", fontsize=numsFontSize,fontweight='bold' )
    #     text = ax4.text(0+0.04, i+0.04, str(modelAndGT_cosSimNM[i,0].round(2)).replace('0.','.'), \
    #                                 ha="center", va="center", color="k", fontsize=numsFontSize, fontweight='bold' )
    #     text = ax4.text(0+0.02, i+0.02, str(modelAndGT_cosSimNM[i,0].round(2)).replace('0.','.'), \
    #                                 ha="center", va="center", color="w", fontsize=numsFontSize, fontweight='bold' )
    
    # #
    
    # ax3 = plt.subplot2grid((1,3),(0,2)) 
    # ax3.imshow(modelPair_cosSimNM, vmin=0, vmax=vmaxx, cmap='jet')
    # ax3.set_xticks(np.arange(0, len(fnames)))
    # ax3.set_xticklabels(np.arange(0, len(fnames)), fontsize=20, fontweight='bold')
    # ax3.set_yticks(np.arange(0, len(fnames)))
    # ax3.set_yticklabels(np.arange(0, len(fnames)), fontsize=20, fontweight='bold')
    # #ax3.set_xlabel('{ \\bf $avg model = \\frac{A+B}{2} $ } ', fontsize=22, fontweight='bold')
    # ax3.set_title( str('Null - without CA matching'), fontsize=22, fontweight='bold', ha="center", va="bottom" )
    # # Loop over data dimensions and create text annotations.
    # for i in range(len(fnames)):
    #     for j in range(len(fnames)):
    #         text = ax3.text(j,      i,      str(modelPair_cosSimNM[i, j].round(2)).replace('0.','.'), \
    #                                         ha="center", va="center", color="k", fontsize=numsFontSize, fontweight='bold' )
    #         text = ax3.text(j+0.04, i+0.04, str(modelPair_cosSimNM[i, j].round(2)).replace('0.','.'), \
    #                                         ha="center", va="center", color="k", fontsize=numsFontSize, fontweight='bold' )
    #         text = ax3.text(j+0.02, i+0.02, str(modelPair_cosSimNM[i, j].round(2)).replace('0.','.'), \
    #                                         ha="center", va="center", color="w", fontsize=numsFontSize, fontweight='bold' )
    # #
    # # #
    # #
    cax = f.add_axes([0.84, 0.16, 0.03, 0.66]) 
    f.colorbar(im, cax=cax)
    cax.set_title( str(r'$ \frac{(A.B)}{ \Vert A \Vert \Vert B \Vert } $'), fontsize=22, fontweight='bold', ha="center", va="bottom" )
    #
    #plt.suptitle( str('{ \\bf '+measureTitle + ' - ' + ftit + '}'), fontsize=24, fontweight='bold' )   
    plt.tight_layout()
    #
    if not os.path.exists( str(plt_save_dir + 'CA_model_matches/') ):
        os.makedirs( str(plt_save_dir + 'CA_model_matches/') )
    #
    plt.savefig( str(plt_save_dir + 'CA_model_matches/' + fname_save + '.' + figSaveFileType ) )
    plt.close() 




    # UNCOMMENT TO BRING BACK..
    # maxx = np.max( numCAsEq2_match + numCAsBtwn2n6_match )
    # #
    # if fnames[0].find('EM_model_data_')>=0:
    # # Synth Data
    #     flabels = [  str( str(i)+'. '+fnames[i][ (fnames[i].find('yMinSW')+8):-4].replace('_',' ').replace('pt','.') ) for i in range(len(fnames)) ]
    #     ftit = fnames[0][(fnames[0].find('EM_model_data_')+14):fnames[0].find('_rand')].replace('_',' ').replace('pt','.')
    # else:
    # # Real Data                                   
    #     flabels = [  str( str(i)+'. '+fnames[i][ fnames[i].find('rand'):-4] ) for i in range(len(fnames)) ]
    #     ftit = fnames[0][(fnames[0].find('LearnedModel_')+13):fnames[0].find('_rand')].replace('_',' ').replace('pt','.')
    # #
    # f = plt.figure( figsize=(20,10) ) # size units in inches
    # plt.rc('font', weight='bold', size=20)
    # #
    # ax1 = plt.subplot2grid((2,2),(0,0)) 
    # ax1.imshow(numCAsEq2_match+numCAsBtwn2n6_match, vmin=0, vmax=maxx, cmap='jet' )
    # ax1.set_xticks(np.arange(0, len(fnames)))
    # ax1.set_yticks(np.arange(0, len(fnames)))
    # ax1.set_yticklabels(flabels, fontsize=14, fontweight='bold', rotation=45)
    # ax1.set_title('2 $\leq$ |CA| $\leq$ 6')
    # #
    # # Loop over data dimensions and create text annotations.
    # for i in range(len(fnames)):
    #     for j in range(len(fnames)):
    #         text = ax1.text(j, i, numCAsEq2_match[i, j]+numCAsBtwn2n6_match[i, j], ha="center", va="center", color="k", fontsize=20, fontweight='bold' )
    #         text = ax1.text(j+0.04, i+0.04, numCAsEq2_match[i, j]+numCAsBtwn2n6_match[i, j], ha="center", va="center", color="k", fontsize=20, fontweight='bold' )
    #         text = ax1.text(j+0.02, i+0.02, numCAsEq2_match[i, j]+numCAsBtwn2n6_match[i, j], ha="center", va="center", color="w", fontsize=20, fontweight='bold' )
    # #
    # ax2 = plt.subplot2grid((2,2),(0,1)) 
    # im = ax2.imshow(numCAsBtwn2n6_match, vmin=0, vmax=maxx, cmap='jet' )
    # ax2.set_title('3 $\leq$ |CA| $\leq$ 6')
    # ax2.set_xticks(np.arange(0, len(fnames)))
    # ax2.set_yticks(np.arange(0, len(fnames)))
    # #                                           #
    # # Loop over data dimensions and create text annotations.
    # for i in range(len(fnames)):
    #     for j in range(len(fnames)):
    #         text = ax2.text(j, i, numCAsBtwn2n6_match[i, j], ha="center", va="center", color="k", fontsize=20, fontweight='bold' )
    #         text = ax2.text(j+0.04, i+0.04, numCAsBtwn2n6_match[i, j], ha="center", va="center", color="k", fontsize=20, fontweight='bold' )
    #         text = ax2.text(j+0.02, i+0.02, numCAsBtwn2n6_match[i, j], ha="center", va="center", color="w", fontsize=20, fontweight='bold' )
    # #





    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



    # ax3 = plt.subplot2grid((2,2),(1,0)) 
    # ax3.imshow(numCAsEq2_matchGT+numCAsBtwn2n6_matchGT, vmin=0, vmax=maxx, cmap='jet' )
    # ax3.set_xticks(np.arange(0, len(fnames)))
    # ax3.set_yticks(np.arange(0, len(fnames)))
    # ax3.set_ylabel('Match with GT')
    # #ax3.set_yticklabels(flabels, fontsize=14, fontweight='bold', rotation=45)
    # ax3.set_title('2 $\leq$ |CA| $\leq$ 6')
    # #
    # # Loop over data dimensions and create text annotations.
    # for i in range(len(fnames)):
    #     for j in range(len(fnames)):
    #         text = ax3.text(j, i, numCAsEq2_matchGT[i, j]+numCAsBtwn2n6_matchGT[i, j], ha="center", va="center", color="k", fontsize=20, fontweight='bold' )
    #         text = ax3.text(j+0.04, i+0.04, numCAsEq2_matchGT[i, j]+numCAsBtwn2n6_matchGT[i, j], ha="center", va="center", color="k", fontsize=20, fontweight='bold' )
    #         text = ax3.text(j+0.02, i+0.02, numCAsEq2_matchGT[i, j]+numCAsBtwn2n6_matchGT[i, j], ha="center", va="center", color="w", fontsize=20, fontweight='bold' )
    # #
    # ax4 = plt.subplot2grid((2,2),(1,1)) 
    # im = ax4.imshow(numCAsBtwn2n6_matchGT, vmin=0, vmax=maxx, cmap='jet' )
    # ax4.set_title('3 $\leq$ |CA| $\leq$ 6')
    # ax4.set_xticks(np.arange(0, len(fnames)))
    # ax4.set_yticks(np.arange(0, len(fnames)))
    # #                                           #
    # # Loop over data dimensions and create text annotations.
    # for i in range(len(fnames)):
    #     for j in range(len(fnames)):
    #         text = ax4.text(j, i, numCAsBtwn2n6_matchGT[i, j], ha="center", va="center", color="k", fontsize=20, fontweight='bold' )
    #         text = ax4.text(j+0.04, i+0.04, numCAsBtwn2n6_matchGT[i, j], ha="center", va="center", color="k", fontsize=20, fontweight='bold' )
    #         text = ax4.text(j+0.02, i+0.02, numCAsBtwn2n6_matchGT[i, j], ha="center", va="center", color="w", fontsize=20, fontweight='bold' )
    # #






    # cax1 = f.add_axes([0.92, 0.15, 0.02, 0.7])
    # f.colorbar(im, cax=cax1)
    # cax1.set_title( str('#CAs/'+str(M)) )
    # #
    # plt.suptitle( str('Varying spike word sampling -  LR' + str(learning_rate) + ' - ' + ftit ), \
    #             fontsize=24, fontweight='bold' )   
    # #
    # if not os.path.exists( str(plt_save_dir + 'CA_model_matches/') ):
    #     os.makedirs( str(plt_save_dir + 'CA_model_matches/') )
    # #
    # plt.savefig( str(plt_save_dir + 'CA_model_matches/' + fname_save + '.pdf' ) )
    # plt.close() 




def scatter_lenSW_inf_vs_obs(nY_obs, pyiEq1_gvnZ_sum):

    print('Note I am not saving this plot in scatter_lenSW_inf_vs_obs')
    maxY = nY_obs.max()
    m,b = np.polyfit( nY_obs, pyiEq1_gvnZ_sum, 1 ) # fit line parameters.
    plt.scatter( nY_obs,  pyiEq1_gvnZ_sum, s=10)
    plt.gca().set_aspect('equal', 'box')
    plt.plot([0, maxY],[0, maxY],'k--')
    plt.plot([0, maxY], [b, b+maxY*m], 'r--')
    plt.text(maxY,0, str('FitLine: y='+str(m.round(2))+'x+'+str(b.round(2))),fontsize=12,fontweight='bold', \
        horizontalalignment='right', verticalalignment='bottom', color='red')
    plt.xlabel(' |Y| obs. ')
    plt.ylabel(' $\sum$ p(Y|Z.inf) ')
    plt.title('Spike Word Length |Y| Inference Bias?')
    plt.grid()
    plt.show()






def plot_QQ_bestFit_dists(yc_sm, yc_sn, yc_nm, y1_sm, y1_sn, y1_nm, y2_sm, y2_sn, y2_nm, mx1, nx1, sx1, mx2, nx2, sx2, mx3, nx3, sx3, \
                                    m1, n1, s1, m2, n2, s2, m3, n3, s3, plt_save_dir, params_str, figSaveFileType):



    f = plt.figure( figsize=(12,8) ) # size units in inches   
    plt.rc('font', weight='bold', size=16)
    

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    ax1 = plt.subplot2grid((2,3),(1,0)) 
    ax1.plot( np.array([0, 1]),np.array([0, 1]), 'k--', linewidth=2 )
    ax1.plot( sx1, mx1, 'bx-', linewidth=3, label='Syn v Mov' )
    ax1.plot( sx1, nx1, 'gx-', linewidth=3, label='Syn v Wnz' )
    #ax1.plot( mx1, nx1, 'c.--', linewidth=2, label='Mov v Wnz' )
    ax1.text( 1, 0.0, str('QQ='+str(yc_sm.round(3))), color='blue', horizontalalignment='right', verticalalignment='bottom', fontweight='bold' )   
    ax1.text( 1, 0.1, str('QQ='+str(yc_sn.round(3))), color='green', horizontalalignment='right', verticalalignment='bottom', fontweight='bold' )  
    ax1.legend(loc='upper left',fontsize=12)
    ax1.set_ylabel('CDF vs CDF (QQ-plots)')
    ax1.set_aspect('equal')
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_yticks([0, 0.5, 1])
    ax1.grid()
    
    #
    ax1b = plt.subplot2grid((2,3),(0,0)) 
    ax1b.plot( s1[1][1:], s1[0]/s1[0].sum(), 'ro-', linewidth=2, label='Syn' )
    ax1b.plot( m1[1][1:], m1[0]/m1[0].sum(), 'bx-', linewidth=2, label='Mov' )
    ax1b.plot( n1[1][1:], n1[0]/n1[0].sum(), 'gx-', linewidth=2, label='Wnz' )
    ax1b.grid()
    ax1b.legend(loc='upper right',fontsize=12)
    ax1b.set_ylabel( 'PDFs' )
    ax1b.set_xlabel(r'# active cells $\vert \vec{y} \vert$')
    ax1b.set_adjustable('box')
    ax1b.set_title( r'$\vert \vec{y} \vert$ - cardinality' )
    #



    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    ax2 = plt.subplot2grid((2,3),(1,1)) 
    ax2.plot( np.array([0, 1]),np.array([0, 1]), 'k--', linewidth=2,  )
    ax2.plot( sx2, mx2, 'bx-', linewidth=3, label='Syn v Mov' )
    ax2.plot( sx2, nx2, 'gx-', linewidth=3, label='Syn v Wnz' )
    #ax2.plot( mx2, nx2, 'c.--', linewidth=3, label='Mov v Wnz' )
    ax2.text( 1, 0.0, str('QQ='+str(y1_sm.round(3))), color='blue', horizontalalignment='right', verticalalignment='bottom', fontweight='bold' )   
    ax2.text( 1, 0.1, str('QQ='+str(y1_sn.round(3))), color='green', horizontalalignment='right', verticalalignment='bottom', fontweight='bold' )  
    ax2.set_aspect('equal')
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_yticks([0, 0.5, 1])
    ax2.grid()
    #
    ax2b = plt.subplot2grid((2,3),(0,1)) 
    ax2b.plot( np.arange(len(s2[0])), s2[0]/s2[0].sum(), 'ro-', linewidth=3, label='Syn' )
    ax2b.plot( np.arange(len(m2[0])), m2[0]/m2[0].sum(), 'bx-', linewidth=3, label='Mov' )
    ax2b.plot( np.arange(len(n2[0])), n2[0]/n2[0].sum(), 'gx-', linewidth=3, label='Wnz' )
    
    ax2b.grid()
    ax2b.set_xlabel('cell ID $y_i$')
    ax2b.set_title( 'single y activity' )
    #


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    ax3 = plt.subplot2grid((2,3),(1,2)) 
    ax3.plot( np.array([0, 1]),np.array([0, 1]), 'k--', linewidth=2 )
    ax3.plot( sx3, mx3, 'bx-', linewidth=3, label='Syn v Mov' )
    ax3.plot( sx3, nx3, 'gx-', linewidth=3, label='Syn vs Wnz' )
    #ax3.plot( mx3, nx3, 'c.--', linewidth=3, label='Mov vs Wnz' )
    ax3.text( 1, 0.0, str('QQ='+str(y2_sm.round(3))), color='blue', horizontalalignment='right', verticalalignment='bottom', fontweight='bold' )   
    ax3.text( 1, 0.1, str('QQ='+str(y2_sn.round(3))), color='green', horizontalalignment='right', verticalalignment='bottom', fontweight='bold' )  
    ax3.set_aspect('equal')
    ax3.set_xticks([0, 0.5, 1])
    ax3.set_yticks([0, 0.5, 1])
    ax3.grid()
    #
    ax3b = plt.subplot2grid((2,3),(0,2)) 
    ax3b.plot( s3[1][1:], s3[0]/s3[0].sum(), 'ro-', linewidth=3, label='Syn' )
    ax3b.plot( m3[1][1:], m3[0]/m3[0].sum(), 'bx-', linewidth=3, label='Mov' )
    ax3b.plot( n3[1][1:], n3[0]/n3[0].sum(), 'gx-', linewidth=3, label='Wnz' )
    ax3b.grid()
    ax3b.set_xlabel(r'$<y_i \cdot y_j>$')
    ax3b.set_title( 'pairwise y coactivity' )
    #
    # plt.suptitle( params_str.replace('_',' ') ) 
    plt.tight_layout()
    plt.savefig( str(plt_save_dir + params_str.replace('.','pt') + '.' + figSaveFileType ) )
    plt.close()
    # plt.show()    

    # MAYBE TODO. PLOT AND FIT DISTRIBUTIONS OF ROWSUMS IN PAIRWISE CORRELATIONS. IF SO, WHERE TO STOP ??
    

