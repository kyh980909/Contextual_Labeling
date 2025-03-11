class Args:
    def __init__(self):
        # 기본 학습 파라미터
        self.lr = 4e-5
        self.lr_backbone_names = ["backbone.0"]
        self.lr_backbone = 4e-6
        self.lr_linear_proj_names = ['reference_points', 'sampling_offsets']
        self.lr_linear_proj_mult = 0.1
        self.batch_size = 1
        self.weight_decay = 1e-4
        self.epochs = 51
        self.lr_drop = 35
        self.lr_drop_epochs = None
        self.clip_max_norm = 0.1
        
        # Deformable DETR variants
        self.with_box_refine = False
        self.two_stage = False
        self.masks = False
        self.backbone = 'dino_resnet50'
        
        # 모델 파라미터
        self.frozen_weights = None
        self.dilation = False
        self.position_embedding = 'sine'
        self.position_embedding_scale = 2 * 3.14159  # 2 * np.pi
        self.num_feature_levels = 4
        
        # Transformer 설정
        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 1024
        self.hidden_dim = 256
        self.dropout = 0.1
        self.nheads = 8
        self.num_queries = 100
        self.dec_n_points = 4
        self.enc_n_points = 4
        
        # Loss 관련
        self.aux_loss = True
        self.set_cost_class = 2
        self.set_cost_bbox = 5
        self.set_cost_giou = 2
        self.cls_loss_coef = 2
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 2
        self.focal_alpha = 0.25
        
        # 데이터셋 파라미터
        self.device = 'cuda'
        self.seed = 42
        self.resume = ''
        self.start_epoch = 0
        self.eval_every = 5
        self.num_workers = 3
        self.output_dir = ''

        # OW-DETR 관련
        self.PREV_INTRODUCED_CLS = 0
        self.CUR_INTRODUCED_CLS = 20
        self.top_unk = 5
        self.featdim = 1024
        self.invalid_cls_logits = False
        self.NC_branch = False
        self.bbox_thresh = 0.3
        self.nc_loss_coef = 2
        self.train_set = 'owod_t1_train'
        self.test_set = 'owod_all_task_test'
        self.num_classes = 81
        self.nc_epoch = 0
        self.dataset = 'TOWOD'
        self.data_root = './data/OWOD'
        self.unk_conf_w = 1.0
        self.unmatched_boxes = False
        
        # PROB OWOD 관련
        self.model_type = 'prob'
        self.wandb_project = 'PROB_OWOD'
        self.obj_loss_coef = 1
        self.obj_temp = 1.0
        self.freeze_prob_model = False
        
        # Exemplar replay 관련
        self.num_inst_per_class = 50
        self.exemplar_replay_selection = False
        self.exemplar_replay_max_length = int(1e10)
        self.exemplar_replay_dir = ''
        self.exemplar_replay_prev_file = ''
        self.exemplar_replay_cur_file = ''
        self.exemplar_replay_random = False

        # Contextual labeling 관련
        self.objectness_thr = 0.6

args = Args()