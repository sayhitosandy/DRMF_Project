'''
Modified on Oct. 10, 2017 
@author: Hao Wu, Zhengxin Zhang
'''

import os
import time
import math
import numpy as np
from util import eval_RATING, eval_RATING_biased
from util import eval_RANKING, eval_RANKING_biased, val_format
from text_analysis.models import CNN_module, CNN_GRU_module

TOP_N = [50, 100, 150, 200, 250, 300]
ENDURE_COUNT = 5
MIN_ITER=5
CONVERG_THRESHOLD = 1e-5
a = 1
b = 0.01
def _DRMF(res_dir, train_user, train_item, valid_user, test_user,
           R, DNN_X, DNN_Y, vocab_size, init_W=None, give_weight=True,
           max_iter=50, lambda_u=1, lambda_v=100, dimension=50,
           dropout_rate=0.2, emb_dim=200, num_kernel_per_ws=50,
           dnn_type='CNN_GRU', reg_schema='Dual', gru_outdim=50, maxlen_doc=[10, 10], maxlen_sent=[30, 30], query_list=[]):
    # explicit setting
    num_user = R.shape[0]
    num_item = R.shape[1]
    PREV_LOSS = 1e-50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    f1 = open(res_dir + '/state.log', 'a')
    f1.write("### _DRMF-%s-%s ###\n\n" % (reg_schema, dnn_type))
    f1.write("===Configuration===\n")
    f1.write("lambda_u=%f, lambda_v=%f\n" % (lambda_u, lambda_v))
    f1.write("maxlen_doc=[%d,%d], maxlen_sent=[%d,%d]\n" % (maxlen_doc[0], maxlen_doc[1], maxlen_sent[0], maxlen_sent[1]))
    f1.write("emb_dim=%d, dimension=%d, num_kernel_per_ws=%d, dropout_rate=%.2f, gru_outdim=%d\n\n" % (emb_dim, dimension, num_kernel_per_ws, dropout_rate, gru_outdim))
    f1.write("Tr:Training, Val:Validation, Te:Test, []: [MAE, MSE, RMSE]\n")

    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    Test_R = test_user[1]
    Valid_R = valid_user[1]

    if give_weight is True:
        item_weight = np.array([math.sqrt(len(i))  for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
        
        user_weight = np.array([math.sqrt(len(u))  for u in Train_R_I], dtype=float)
        user_weight = (float(num_user) / user_weight.sum()) * user_weight
        
    else:
        item_weight = np.ones(num_item, dtype=float)
        user_weight = np.ones(num_user, dtype=float)

    pre_val_rmse = 1e10
    if dnn_type == 'CNN':
        if reg_schema == 'Item' or reg_schema == 'Dual':
            dnn_module_x = CNN_module(dimension, vocab_size, dropout_rate, emb_dim, maxlen_doc[0] * maxlen_sent[0], num_kernel_per_ws, init_W)
        if reg_schema == 'User' or reg_schema == 'Dual':
            dnn_module_y = CNN_module(dimension, vocab_size, dropout_rate, emb_dim, maxlen_doc[1] * maxlen_sent[1], num_kernel_per_ws, init_W)
            
    if dnn_type == 'CNN_GRU':
        if reg_schema == 'Item' or reg_schema == 'Dual':
            dnn_module_x = CNN_GRU_module(dimension, vocab_size, dropout_rate, emb_dim, gru_outdim, maxlen_doc[0], maxlen_sent[0], num_kernel_per_ws, init_W)
        if reg_schema == 'User' or reg_schema == 'Dual':
            dnn_module_y = CNN_GRU_module(dimension, vocab_size, dropout_rate, emb_dim, gru_outdim, maxlen_doc[1], maxlen_sent[1], num_kernel_per_ws, init_W)

    if reg_schema == 'Item' or reg_schema == 'Dual':  
        theta = dnn_module_x.get_projection_layer(DNN_X)
    if reg_schema == 'User' or reg_schema == 'Dual':
        phi = dnn_module_y.get_projection_layer(DNN_Y)
    
    
    np.random.seed(133)
    
    if reg_schema == 'User' or reg_schema == 'Dual': 
        U = phi
    else:
        U = np.random.uniform(size=(num_user, dimension))
        
    if reg_schema == 'Item' or reg_schema == 'Dual':     
        V = theta
    else:
        V = np.random.uniform(size=(num_item, dimension))
        
    count = 0
    prev_iter=0
    for iteration in range(max_iter):
        loss = 0
        tic = time.time()
        print ("%d iteration\t(patience: %d)" % (iteration, count))
        f1.write("%d iteration\t(patience: %d)\n" % (iteration, count))

        VV = b * (V.T.dot(V))
        sub_loss = np.zeros(num_user)
        # update U
        for i in range(num_user):
            idx_item = train_user[0][i]
            V_i = V[idx_item]
            R_i = Train_R_I[i]
            
            tmp_A = VV + (a - b) * (V_i.T.dot(V_i))
            A = tmp_A + lambda_u * user_weight[i] * np.eye(dimension)
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0) 
            if reg_schema == 'User' or reg_schema == 'Dual':
                B = B + lambda_u * user_weight[i] * phi[i]    
            U[i] = np.linalg.solve(A, B)
            # -\frac{\lambda_u}{2}\sum_i u_i^Tu_i
            if reg_schema == 'Item':
                sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])
                
        loss += np.sum(sub_loss)

        sub_loss_dev = np.zeros(num_item)
        sub_loss = np.zeros(num_item)
        # update V
        UU = b * (U.T.dot(U))
        for j in range(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
            A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
            B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)).sum(0)
            if reg_schema == 'Item' or reg_schema == 'Dual':
                B = B + lambda_v * item_weight[j] * theta[j]
               
            V[j] = np.linalg.solve(A, B)
            # -\sum_i\sum_j\frac{c_{i,j}}{2}(r_{ij}-u_i^T v_j)^2
            sub_loss_dev[j] = -0.5 * np.square(R_j * a).sum()
            sub_loss_dev[j] += a * np.sum((U_j.dot(V[j])) * R_j)
            sub_loss_dev[j] += -0.5 * np.dot(V[j].dot(tmp_A), V[j])
            # -\frac{\lambda_v}{2}\sum_jv_j^Tv_j
            if reg_schema == 'User':
                sub_loss[j] = -0.5 * lambda_v * np.dot(V[j], V[j])
                
        loss += np.sum(sub_loss_dev)
        loss += np.sum(sub_loss)
        
        seed = np.random.randint(100000)
        
        if reg_schema == 'Item' or reg_schema == 'Dual':  
            history_x = dnn_module_x.train(DNN_X, V, item_weight, seed)
            theta = dnn_module_x.get_projection_layer(DNN_X)
            # -\frac{\lambda_v}{2}\sum_j(v_j-\theta_j)^T(v_j-\theta_j)
            cnn_loss_x = history_x.history['loss'][-1]
            loss += -0.5 * lambda_v * cnn_loss_x * num_item
            
        if reg_schema == 'User' or reg_schema == 'Dual':
            history_y = dnn_module_y.train(DNN_Y, U, user_weight, seed) 
            phi = dnn_module_y.get_projection_layer(DNN_Y)
            # -\frac{\lambda_u}{2}\sum_i (u_i-\phi_i)^T(u_i-\phi_i)
            cnn_loss_y = history_y.history['loss'][-1]
            loss += -0.5 * lambda_u * cnn_loss_y * num_user

        # tr_mae, tr_mse, tr_rmse = eval_RATING(Train_R_I, U, V, train_user[0])
        val_mae, val_mse, val_rmse = eval_RATING(Valid_R, U, V, valid_user[0])
        te_mae, te_mse, te_rmse = eval_RATING(Test_R, U, V, test_user[0])

        toc = time.time()
        elapsed = toc - tic

        if iteration == 0:
            converge = -1
        else: 
            converge = abs((loss - PREV_LOSS) / PREV_LOSS)
        
#         if (val_rmse < pre_val_rmse):
#             if dnn_type == 'CNN':
#                 if os.path.exists(res_dir + '/drmf_cnn') is not True:
#                     os.mkdir(res_dir + '/drmf_cnn')
#                     if os.path.exists(res_dir + '/drmf_cnn/dual') is not True:
#                         os.mkdir(res_dir + '/drmf_cnn/dual')
#                     if os.path.exists(res_dir + '/drmf_cnn/user') is not True:
#                         os.mkdir(res_dir + '/drmf_cnn/user')
#                     if os.path.exists(res_dir + '/drmf_cnn/item') is not True:
#                         os.mkdir(res_dir + '/drmf_cnn/item')
#                 if reg_schema == 'Dual':
#                     np.savetxt(res_dir + '/drmf_cnn/dual/U.dat', U)
#                     np.savetxt(res_dir + '/drmf_cnn/dual/V.dat', V)
#                     np.savetxt(res_dir + '/drmf_cnn/dual/theta.dat', theta)
#                     np.savetxt(res_dir + '/drmf_cnn/dual/phi.dat', phi)
#                     dnn_module_x.save_model(res_dir + '/drmf_cnn/dual/x_weights.hdf5')
#                     dnn_module_y.save_model(res_dir + '/drmf_cnn/dual/y_weights.hdf5')
#                 if reg_schema == 'User':
#                     np.savetxt(res_dir + '/drmf_cnn/user/U.dat', U)
#                     np.savetxt(res_dir + '/drmf_cnn/user/V.dat', V)
#                     np.savetxt(res_dir + '/drmf_cnn/user/phi.dat', phi)
#                     dnn_module_y.save_model(res_dir + '/drmf_cnn/user/y_weights.hdf5')
#                 if reg_schema == 'Item':
#                     np.savetxt(res_dir + '/drmf_cnn/item/U.dat', U)
#                     np.savetxt(res_dir + '/drmf_cnn/item/V.dat', V)
#                     np.savetxt(res_dir + '/drmf_cnn/item/theta.dat', theta)
#                     dnn_module_x.save_model(res_dir + '/drmf_cnn/item/x_weights.hdf5')
#             if dnn_type == 'CNN_GRU':
#                 if os.path.exists(res_dir + '/drmf_cnn_gru') is not True:
#                     os.mkdir(res_dir + '/drmf_cnn_gru')
#                     if os.path.exists(res_dir + '/drmf_cnn_gru/dual') is not True:
#                         os.mkdir(res_dir + '/drmf_cnn_gru/dual')
#                     if os.path.exists(res_dir + '/drmf_cnn_gru/user') is not True:
#                         os.mkdir(res_dir + '/drmf_cnn_gru/user')
#                     if os.path.exists(res_dir + '/drmf_cnn_gru/item') is not True:
#                         os.mkdir(res_dir + '/drmf_cnn_gru/item')
#                 if reg_schema == 'Dual':
#                     np.savetxt(res_dir + '/drmf_cnn_gru/dual/U.dat', U)
#                     np.savetxt(res_dir + '/drmf_cnn_gru/dual/V.dat', V)
#                     np.savetxt(res_dir + '/drmf_cnn_gru/dual/theta.dat', theta)
#                     np.savetxt(res_dir + '/drmf_cnn_gru/dual/phi.dat', phi)
#                     dnn_module_x.save_model(res_dir + '/drmf_cnn_gru/dual/x_weights.hdf5')
#                     dnn_module_y.save_model(res_dir + '/drmf_cnn_gru/dual/y_weights.hdf5')
#                 if reg_schema == 'User':
#                     np.savetxt(res_dir + '/drmf_cnn_gru/user/U.dat', U)
#                     np.savetxt(res_dir + '/drmf_cnn_gru/user/V.dat', V)
#                     np.savetxt(res_dir + '/drmf_cnn_gru/user/phi.dat', phi)
#                     dnn_module_y.save_model(res_dir + '/drmf_cnn_gru/user/y_weights.hdf5')
#                 if reg_schema == 'Item':
#                     np.savetxt(res_dir + '/drmf_cnn_gru/item/U.dat', U)
#                     np.savetxt(res_dir + '/drmf_cnn_gru/item/V.dat', V)
#                     np.savetxt(res_dir + '/drmf_cnn_gru/item/theta.dat', theta)
#                     dnn_module_x.save_model(res_dir + '/drmf_cnn_gru/item/x_weights.hdf5')
#         else:
#             count = count + 1
        # for fast running, without saving models
        if (val_rmse >= pre_val_rmse):
            count = count + 1
        pre_val_rmse = val_rmse

        print ("Loss: %.5f Elpased: %.4fs Converge: %.6f Val: [%.5f, %.5f, %.5f] Te: [%.5f, %.5f, %.5f] " % (
            loss, elapsed, converge, val_mae, val_mse, val_rmse, te_mae, te_mse, te_rmse))
        f1.write("Loss: %.5f Elpased: %.4fs Converge: %.6f  Val: [%.5f, %.5f, %.5f] Te: [%.5f, %.5f, %.5f]\n" % (
            loss, elapsed, converge, val_mae, val_mse, val_rmse, te_mae, te_mse, te_rmse))

        
        PREV_LOSS = loss
        prev_iter=iteration+1
        '''evaluation for top-n recommendation'''
        if (iteration>=MIN_ITER and ( converge <= CONVERG_THRESHOLD or count >= ENDURE_COUNT or iteration >= max_iter)):
            tic = time.time()
            _pre, _rec, _ndcg, _map = eval_RANKING(U, V, query_list, TOP_N)
            toc = time.time()
            elapsed = toc - tic
            print ("Elpased: %.4fs, Top-%s, Precision: %s Recall: %s NDCG: %s MAP: %s\n" % (elapsed, TOP_N, val_format(_pre, ".5"), val_format(_rec, ".5"), val_format(_ndcg, ".5"), val_format(_map, ".5")))
            f1.write("Elpased: %.4fs, Top-%s, Precision: %s Recall: %s NDCG: %s MAP: %s\n" % (elapsed, TOP_N, val_format(_pre, ".5"), val_format(_rec, ".5"), val_format(_ndcg, ".5"), val_format(_map, ".5")))
            break

    '''Tuning DRMF with fixed theta and phi'''
#     for iteration in range(max_iter):
#         loss = 0
#         tic = time.time()
#         print ("%d iteration\t(patience: %d)" % (iteration + prev_iter, count))
#         f1.write("%d iteration\t(patience: %d)\n" % (iteration + prev_iter, count))
# 
#         VV = b * (V.T.dot(V))
#         sub_loss = np.zeros(num_user)
#         # update U
#         for i in range(num_user):
#             idx_item = train_user[0][i]
#             V_i = V[idx_item]
#             R_i = Train_R_I[i]
#             
#             tmp_A = VV + (a - b) * (V_i.T.dot(V_i))
#             A = tmp_A + lambda_u * user_weight[i] * np.eye(dimension)
#             B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0) 
#             if reg_schema == 'User' or reg_schema == 'Dual':
#                 B = B + lambda_u * user_weight[i] * phi[i]    
#             U[i] = np.linalg.solve(A, B)
#             # -\frac{\lambda_u}{2}\sum_i u_i^Tu_i
#             if reg_schema == 'Item':
#                 sub_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])
#                 
#         loss += np.sum(sub_loss)
# 
#         sub_loss_dev = np.zeros(num_item)
#         sub_loss = np.zeros(num_item)
#         # update V
#         UU = b * (U.T.dot(U))
#         for j in range(num_item):
#             idx_user = train_item[0][j]
#             U_j = U[idx_user]
#             R_j = Train_R_J[j]
# 
#             tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
#             A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
#             B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)).sum(0)
#             if reg_schema == 'Item' or reg_schema == 'Dual':
#                 B = B + lambda_v * item_weight[j] * theta[j]
#                
#             V[j] = np.linalg.solve(A, B)
#             # -\sum_i\sum_j\frac{c_{i,j}}{2}(r_{ij}-u_i^T v_j)^2
#             sub_loss_dev[j] = -0.5 * np.square(R_j * a).sum()
#             sub_loss_dev[j] += a * np.sum((U_j.dot(V[j])) * R_j)
#             sub_loss_dev[j] += -0.5 * np.dot(V[j].dot(tmp_A), V[j])
#             # -\frac{\lambda_v}{2}\sum_jv_j^Tv_j
#             if reg_schema == 'User':
#                 sub_loss[j] = -0.5 * lambda_v * np.dot(V[j], V[j])
#                 
#         loss += np.sum(sub_loss_dev)
#         loss += np.sum(sub_loss)
#         
#         if reg_schema == 'Item' or reg_schema == 'Dual':  
#             # -\frac{\lambda_v}{2}\sum_j(v_j-\theta_j)^T(v_j-\theta_j)
#             for j in range(num_item):
#                 epsilon_j = np.subtract(V[j], theta[j])
#                 loss += -0.5 * lambda_v * epsilon_j.T.dot(epsilon_j) 
#             
#         if reg_schema == 'User' or reg_schema == 'Dual':
#             # -\frac{\lambda_u}{2}\sum_i (u_i-\phi_i)^T(u_i-\phi_i)
#             for i in range(num_user):
#                 epsilon_i = np.subtract(U[i], phi[i])
#                 loss += -0.5 * lambda_u * epsilon_i.T.dot(epsilon_i)
# 
#         # tr_mae, tr_mse, tr_rmse = eval_RATING(Train_R_I, U, V, train_user[0])
#         val_mae, val_mse, val_rmse = eval_RATING(Valid_R, U, V, valid_user[0])
#         te_mae, te_mse, te_rmse = eval_RATING(Test_R, U, V, test_user[0])
# 
#         toc = time.time()
#         elapsed = toc - tic
#         converge = abs((loss - PREV_LOSS) / PREV_LOSS)
#         
#         # for fast running, without saving models
#         if (val_rmse >= pre_val_rmse):
#             count = count + 1
#         pre_val_rmse = val_rmse
# 
#         print ("Loss: %.5f Elpased: %.4fs Converge: %.6f Val: [%.5f, %.5f, %.5f] Te: [%.5f, %.5f, %.5f] " % (
#             loss, elapsed, converge, val_mae, val_mse, val_rmse, te_mae, te_mse, te_rmse))
#         f1.write("Loss: %.5f Elpased: %.4fs Converge: %.6f  Val: [%.5f, %.5f, %.5f] Te: [%.5f, %.5f, %.5f]\n" % (
#             loss, elapsed, converge, val_mae, val_mse, val_rmse, te_mae, te_mse, te_rmse))
# 
#         '''evaluation for top-n recommendation'''
#         if (converge <= CONVERG_THRESHOLD or iteration >= max_iter):
#             tic = time.time()
#             _pre, _rec, _ndcg, _map = eval_RANKING(U, V, query_list, TOP_N)
#             toc = time.time()
#             elapsed = toc - tic
#             print ("Elpased: %.4fs, Top-%s, Precision: %s Recall: %s NDCG: %s MAP: %s\n" % (elapsed, TOP_N, val_format(_pre, ".5"), val_format(_rec, ".5"), val_format(_ndcg, ".5"), val_format(_map, ".5")))
#             f1.write("Elpased: %.4fs, Top-%s, Precision: %s Recall: %s NDCG: %s MAP: %s\n" % (elapsed, TOP_N, val_format(_pre, ".5"), val_format(_rec, ".5"), val_format(_ndcg, ".5"), val_format(_map, ".5")))
#             break
# 
#         PREV_LOSS = loss

    f1.close()
    

def _PMF(res_dir, train_user, train_item, valid_user, test_user,
           R, max_iter=50, lambda_u=1, lambda_v=100, dimension=50, give_weight=False, query_list=[]):
    # explicit setting

    num_user = R.shape[0]
    num_item = R.shape[1]
    PREV_LOSS = 1e-50
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    f1 = open(res_dir + '/state.log', 'a')
    f1.write("### _PMF ###\n\n")
    f1.write("===Configuration===\n")
    f1.write("lambda_u=%f, lambda_v=%f\n\n" % (lambda_u, lambda_v))
    f1.write("Tr:Training, Val:Validation, Te:Test, []: [MAE, MSE, RMSE]\n")

    Train_R_I = train_user[1]
    Train_R_J = train_item[1]
    Test_R = test_user[1]
    Valid_R = valid_user[1]

    if give_weight is True:
        item_weight = np.array([math.sqrt(len(i))
                                for i in Train_R_J], dtype=float)
        item_weight = (float(num_item) / item_weight.sum()) * item_weight
        
        user_weight = np.array([math.sqrt(len(u))
                                for u in Train_R_I], dtype=float)
        user_weight = (float(num_user) / user_weight.sum()) * user_weight
        
    else:
        item_weight = np.ones(num_item, dtype=float)
        user_weight = np.ones(num_user, dtype=float)

    pre_val_rmse = 1e10

    
    np.random.seed(133)
    U = np.random.uniform(size=(num_user, dimension))
    V = np.random.uniform(size=(num_item, dimension))

    count = 0
    for iteration in range(max_iter):
        loss = 0
        tic = time.time()
        print ("%d iteration\t(patience: %d)" % (iteration, count))
        f1.write("%d iteration\t(patience: %d)\n" % (iteration, count))
        VV = b * (V.T.dot(V))
        sub_loss = np.zeros(num_user)
        # update U
        for i in range(num_user):
            idx_item = train_user[0][i]
            V_i = V[idx_item]
            R_i = Train_R_I[i]
            
            tmp_A = VV + (a - b) * (V_i.T.dot(V_i))
            A = tmp_A + lambda_u * user_weight[i] * np.eye(dimension)
            B = (a * V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0)
            U[i] = np.linalg.solve(A, B)
            
            sub_loss[i] = np.dot(U[i], U[i])
        loss += -0.5 * lambda_u * np.sum(sub_loss)

        sub_loss = np.zeros(num_item)
        # update V
        UU = b * (U.T.dot(U))
        for j in range(num_item):
            idx_user = train_item[0][j]
            U_j = U[idx_user]
            R_j = Train_R_J[j]

            tmp_A = UU + (a - b) * (U_j.T.dot(U_j))
            A = tmp_A + lambda_v * item_weight[j] * np.eye(dimension)
            B = (a * U_j * (np.tile(R_j, (dimension, 1)).T)).sum(0)
            V[j] = np.linalg.solve(A, B)

            sub_loss[j] = -0.5 * np.square(R_j * a).sum()
            sub_loss[j] += a * np.sum((U_j.dot(V[j])) * R_j)
            sub_loss[j] += -0.5 * np.dot(V[j].dot(tmp_A), V[j])
            sub_loss[j] += -0.5 * lambda_v * np.dot(V[j], V[j])
        loss += np.sum(sub_loss)

        # tr_mae, tr_mse, tr_rmse = eval_RATING(Train_R_I, U, V, train_user[0])
        val_mae, val_mse, val_rmse = eval_RATING(Valid_R, U, V, valid_user[0])
        te_mae, te_mse, te_rmse = eval_RATING(Test_R, U, V, test_user[0])

        toc = time.time()
        elapsed = toc - tic

        if iteration == 0:
            converge = -1
        else:
            converge = abs((loss - PREV_LOSS) / PREV_LOSS)

#         if (val_rmse < pre_val_rmse):
#             if os.path.exists(res_dir + '/pmf') is not True:
#                     os.mkdir(res_dir + '/pmf')
#             np.savetxt(res_dir + '/pmf/U.dat', U)
#             np.savetxt(res_dir + '/pmf/V.dat', V)
#         else:
#             count = count + 1
        # for fast running, without saving models
        if (val_rmse >= pre_val_rmse):
            count = count + 1
        
        pre_val_rmse = val_rmse

        print ("Loss: %.5f Elpased: %.4fs Converge: %.6f Val: [%.5f, %.5f, %.5f] Te: [%.5f, %.5f, %.5f] " % (
            loss, elapsed, converge, val_mae, val_mse, val_rmse, te_mae, te_mse, te_rmse))
        f1.write("Loss: %.5f Elpased: %.4fs Converge: %.6f  Val: [%.5f, %.5f, %.5f] Te: [%.5f, %.5f, %.5f]\n" % (
            loss, elapsed, converge, val_mae, val_mse, val_rmse, te_mae, te_mse, te_rmse))
        
        '''evaluation for top-n recommendation'''
        if (iteration>=MIN_ITER and ( converge <= CONVERG_THRESHOLD or count >= ENDURE_COUNT or iteration >= max_iter)):
        # if True:
            tic = time.time()
            _pre, _rec, _ndcg, _map = eval_RANKING(U, V, query_list, TOP_N)
            toc = time.time()
            elapsed = toc - tic
            print ("Elpased: %.4fs, Top-%s, Precision: %s Recall: %s NDCG: %s MAP: %s\n" % (elapsed, TOP_N, val_format(_pre, ".5"), val_format(_rec, ".5"), val_format(_ndcg, ".5"), val_format(_map, ".5")))
            f1.write("Elpased: %.4fs, Top-%s, Precision: %s Recall: %s NDCG: %s MAP: %s\n" % (elapsed, TOP_N, val_format(_pre, ".5"), val_format(_rec, ".5"), val_format(_ndcg, ".5"), val_format(_map, ".5")))
            break
        
        PREV_LOSS = loss

    f1.close()


    


