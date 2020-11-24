from model.Prediction.trajPredEngine import TrajPredEngine
import torch
import datetime
import numpy as np

MANUAL_BREAK = True

class TraphicEngine(TrajPredEngine):
    """
    Implementation of abstractEngine for traphic
    TODO:maneuver metrics, too much duplicate code with socialEngine
    """

   # def __init__(self, net, optim, train_loader, val_loader, args):
   #     super().__init__(net, optim, train_loader, val_loader, args)
   #     self.save_name = "traphic"

    def __init__(self, net, optim, args):
        super().__init__(net, optim, args)
        self.save_name = "traphic"

    def netPred(self, batch):
        hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc, fut, op_mask, b, d, v, f = batch

        #print('call netPred')
        hist_shape = hist.shape
        #print('hist.shape {}'.format(hist_shape))

        if hist_shape[0] != 25 or hist_shape[1] != 10 or hist_shape[2] != 2:
            print('hist shape changes!!!!!!!!!!!!!!!!!!!!')
            print('hist.shape {}'.format(hist_shape))
        #else:
        #    print('hist shape ok')

        hist_single_rows = hist_shape[1]

        fake_dup_points = 15

        if MANUAL_BREAK and hist_single_rows > 5:
            hist_np = hist.numpy()
            if fake_dup_points < 10:
                fake_array = [hist_np[1, fake_dup_points - 1, :],] * fake_dup_points
                #print('fake_array {}'.format(fake_array))
                hist_np[1, fake_dup_points:, :] = fake_array
            elif fake_dup_points == 10:
                fake_array = [hist_np[0, fake_dup_points - 1, :],] * fake_dup_points
                hist_np[1, :, :] = fake_array
            else:
                fake_dup_points -= 10
                dup_num = min(fake_dup_points, hist_single_rows - 5)
                fake_array = [hist_np[0, fake_dup_points - 1, :],] * dup_num
                print('hist_np 0 {}'.format(hist_np[0, :, :]))
                print('fake_array {}'.format(fake_array))
                hist_np[0, fake_dup_points:, :] = fake_array
                #fake_array = [hist_np[0, fake_dup_points - 1, :],] * 10
                #hist_np[1, :, :] = fake_array


        #print('hist {}'.format(hist))

        if self.args['cuda']:
            hist = hist.cuda(self.device)
            nbrs = nbrs.cuda(self.device)
            upp_nbrs = upp_nbrs.cuda(self.device)
            mask = mask.cuda(self.device)
            upp_mask = upp_mask.cuda(self.device)
            lat_enc = lat_enc.cuda(self.device)
            lon_enc = lon_enc.cuda(self.device)
            fut = fut.cuda(self.device)
            op_mask = op_mask.cuda(self.device)

        fut_pred  = self.net(hist, upp_nbrs, nbrs, upp_mask, mask, lat_enc, lon_enc)

        #print('fut_pred {}'.format(fut_pred.shape))
        #print('fut_pred {}'.format(fut_pred))

        return fut_pred

