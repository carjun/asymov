import argparse
import os
import yaml, pprint, json

from viz import naive_reconstruction_no_rep, naive_reconstruction, very_naive_reconstruction, ground_truth_construction

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--data_dir',
                        help='path to data directory from repo root',
                        type=str)
    parser.add_argument('--data_name',
                        help='which version of the dataset, subset or not',
                        default='xyz',
                        type=str)
    parser.add_argument('--data_splits',
                        help='which splits of the dataset to reconstruct',
                        nargs='*',
                        type=str)

    parser.add_argument('--log_dir',
						help='path to directory to store logs (kit_logs) directory',
						type=str)
    parser.add_argument('--log_ver',
                        help='version in kitml_logs',
                        type=str)

    parser.add_argument('--use_raw',
                        required=True,
                        help='whether to use raw skeleton for clustering',
                        type=int)

    parser.add_argument('--frames_dir',
						help='path to directory to store reconstructions',
						type=str)

    args, _ = parser.parse_known_args()
    with open(args.cfg, 'r') as stream:
        ldd = yaml.safe_load(stream)

    if args.data_dir:
        ldd["PRETRAIN"]["DATA"]["DATA_DIR"] = args.data_dir
    ldd["PRETRAIN"]["DATA"]["DATA_NAME"] = args.data_name
    ldd["PRETRAIN"]["DATA"]["DATA_SPLITS"] = args.data_splits
    
    ldd["CLUSTER"]["USE_RAW"] = args.use_raw

    if args.log_dir:
        ldd["PRETRAIN"]["TRAINER"]["LOG_DIR"] = args.log_dir
    if args.log_ver:
        ldd["CLUSTER"]["VERSION"] = str(args.log_ver)
    else:
        ldd["CLUSTER"]["VERSION"] = sorted([f.name for f in os.scandir(os.path.join(args.log_dir, ldd["CLUSTER"]["CKPT"])) if f.is_dir()], reverse=True)[0]
    
    ldd["FRAMES_DIR"] = args.frames_dir
    ldd["SK_TYPE"] = 'kitml'
    pprint.pprint(ldd)
    return ldd

def main():

    args = parse_args()

    # seq_names = ["00017","00018","00002","00014","00005","00010"]
    seq_names = seq_names = ['01699', #'02855', 
       '00826', '02031', '01920', '02664', '01834',
       '02859', '00398', '03886', '01302', '02053', '00898', '03592',
       '03580', '00771', '01498', '00462', '01292', '02441', '03393',
       '00376', '02149', '03200', '03052', '01788', '00514', '01744',
       '02977', '00243', '02874', '00396', '03597', '02654', '03703',
       '00456', '00812', '00979', '00724', '01443', '03401', '00548',
       '00905', '00835', #'02612', 
       '02388', '03788', '03870', '03181',
       '00199']
    
    if args["PRETRAIN"]["DATA"]["DATA_SPLITS"] != None:
        with open(os.path.join(args["PRETRAIN"]["DATA"]["DATA_DIR"], args["PRETRAIN"]["DATA"]["DATA_NAME"] + '_data_split.json'), 'r') as handle:
            data_split = json.load(handle)
        seq_names = []
        for split in args["PRETRAIN"]["DATA"]["DATA_SPLITS"]:
            seq_names.extend(data_split[split])
    
    data_path = os.path.join(args["PRETRAIN"]["DATA"]["DATA_DIR"], args["PRETRAIN"]["DATA"]["DATA_NAME"] + '_data.pkl')
    if args["CLUSTER"]["USE_RAW"] :
        log_dir = os.path.join(args["PRETRAIN"]["TRAINER"]["LOG_DIR"], 'raw')
    else :
        log_dir = os.path.join(args["PRETRAIN"]["TRAINER"]["LOG_DIR"], args["NAME"], args["CLUSTER"]["VERSION"])
    frame2cluster_mapping_path = os.path.join(log_dir, 'advanced_tr_res_150.pkl')
    contiguous_frame2cluster_mapping_path = os.path.join(log_dir, 'advanced_tr_150.pkl')
    cluster2keypoint_mapping_path = os.path.join(log_dir, 'proxy_centers_tr_150.pkl')
    cluster2frame_mapping_path = os.path.join(log_dir, 'proxy_centers_tr_complete_150.pkl')

    
    if args["FRAMES_DIR"] == None:
        #No filter
        very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, data_path, frame2cluster_mapping_path, cluster2keypoint_mapping_path, args["SK_TYPE"])
        naive_mpjpe_mean = naive_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"])
        naive_no_rep_mpjpe_mean = naive_reconstruction_no_rep(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"])
        #uniform filter
        very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, data_path, frame2cluster_mapping_path, cluster2keypoint_mapping_path, args["SK_TYPE"], filter = 'uniform')
        naive_mpjpe_mean = naive_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='uniform')
        naive_no_rep_mpjpe_mean = naive_reconstruction_no_rep(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='uniform')
        #spline filter
        very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, data_path, frame2cluster_mapping_path, cluster2keypoint_mapping_path, args["SK_TYPE"], filter = 'spline')
        naive_mpjpe_mean = naive_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='spline')
        naive_no_rep_mpjpe_mean = naive_reconstruction_no_rep(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='spline')
    
    else:
        #No filter
        very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, data_path, frame2cluster_mapping_path, cluster2keypoint_mapping_path, args["SK_TYPE"], frames_dir=args["FRAMES_DIR"]+'very_naive')
        naive_mpjpe_mean = naive_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], frames_dir=args["FRAMES_DIR"]+'naive')
        naive_no_rep_mpjpe_mean = naive_reconstruction_no_rep(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], frames_dir=args["FRAMES_DIR"]+'naive_no_rep')
        #uniform filter
        very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, data_path, frame2cluster_mapping_path, cluster2keypoint_mapping_path, args["SK_TYPE"], filter = 'uniform', frames_dir=args["FRAMES_DIR"]+'very_naive_ufilter')
        naive_mpjpe_mean = naive_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='uniform', frames_dir=args["FRAMES_DIR"]+'naive_ufilter')
        naive_no_rep_mpjpe_mean = naive_reconstruction_no_rep(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='uniform', frames_dir=args["FRAMES_DIR"]+'naive_no_rep_ufilter')
        #spline filter
        very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, data_path, frame2cluster_mapping_path, cluster2keypoint_mapping_path, args["SK_TYPE"], filter = 'spline', frames_dir=args["FRAMES_DIR"]+'very_naive_sfilter')
        naive_mpjpe_mean = naive_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='spline', frames_dir=args["FRAMES_DIR"]+'naive_sfilter')
        naive_no_rep_mpjpe_mean = naive_reconstruction_no_rep(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='spline', frames_dir=args["FRAMES_DIR"]+'naive_no_rep_sfilter')
        #original video
        ground_truth_construction(seq_names, data_path, args["SK_TYPE"], frames_dir=args["FRAMES_DIR"]+'ground')

    #No filter
    print('very naive mpjpe : ', very_naive_mpjpe_mean)
    print('naive mpjpe : ', naive_mpjpe_mean)
    print('naive (no rep) mpjpe : ', naive_no_rep_mpjpe_mean)
    print('----------------------------------------------------')
    #uniform filter
    print('uniform filtered very naive mpjpe : ', very_naive_mpjpe_mean)
    print('uniform filtered naive mpjpe : ', naive_mpjpe_mean)
    print('uniform filtered naive (no rep) mpjpe : ', naive_no_rep_mpjpe_mean)
    print('----------------------------------------------------')
    #spline filter
    print('spline filtered very naive mpjpe : ', very_naive_mpjpe_mean)
    print('spline filtered naive mpjpe : ', naive_mpjpe_mean)
    print('spline filtered naive (no rep) mpjpe : ', naive_no_rep_mpjpe_mean)

if __name__ == '__main__':
    main()