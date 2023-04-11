config_dict = {
    'rep':{
        '500kb':{
            'd_in_nfeats':1,'d_in_efeats':2,'d_h_nfeats':32,'d_h_efeats':8,'n_layers':1, 'batch_norm':False,'model_path':'models/500kb_rep.pkl'
        },
        '50kb':{
            'd_in_nfeats':1,'d_in_efeats':2,'d_h_nfeats':32,'d_h_efeats':8,'n_layers':1, 'batch_norm':True,'model_path':'models/50kb_rep.pkl'
        },
        '10kb':{
            'd_in_nfeats':1,'d_in_efeats':2,'d_h_nfeats':32,'d_h_efeats':8,'n_layers':1, 'batch_norm':False,'model_path':'models/10kb_rep.pkl'
        }
    },
    'dci':{
        '500kb':{
            'd_in_nfeats':1,'d_in_efeats':2,'d_h_nfeats':32,'d_h_efeats':8,'n_layers':1, 'batch_norm':False,'model_path':'models/500kb_dci.pkl','sigma':0.8
        },
        '50kb':{
            'd_in_nfeats':1,'d_in_efeats':2,'d_h_nfeats':32,'d_h_efeats':8,'n_layers':1, 'batch_norm':False,'model_path':'models/50kb_dci.pkl','sigma':1
        },
        '10kb':{
            'd_in_nfeats':1,'d_in_efeats':2,'d_h_nfeats':32,'d_h_efeats':8,'n_layers':1, 'batch_norm':True,'model_path':'models/10kb_dci.pkl','sigma':1
        }
    }
}