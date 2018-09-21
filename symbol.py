symbol_list= [ "\\sigma_1_1", "(_1_1",       "\\sum_1_1",   "1_1_1",       "n_1_1",       "2_1_1",       ")_1_1",       "r_1_1",
 "i_2_1",       "\\theta_1_1", "\\sum_2_bot", "b_1_1",       "c_1_1",       "4_1_1",       "3_1_1",       "d_1_1",
 "a_1_1",       "8_1_1",       "7_1_1",       "4_2_nose",    "y_1_1",       "0_1_1",       "y_2_flower",  "x_2_left",
 "x_1_1",       "\\sqrt_1_1",  "L_1_1",       "u_1_1",       "\\mu_1_1",    "k_1_1",       "\\lt_1_1",
 "p_1_1",       "p_2_ear",     "q_1_1",       "j_2_1",       "f_2_cobra",   "\\{_1_1",     "\\}_1_1",     "]_1_1",
 "9_1_1",       "h_1_1",       "\\int_1_1",   "t_2_tail",    "e_1_1",       "z_1_1",       "g_1_1",       "s_1_1",
 "5_2_hook",    "6_1_1",       "v_1_1",       "5_1_1",       "w_1_1",       "\\gt_1_1",    "\\alpha_1_1",
 "\\beta_1_1",  "\\gamma_1_1", "m_1_1",       "l_1_1",       "[_1_1",       "\\infty_1_1", "/_1_1"]



cnt = 0
for i in symbol_list:
 print ("%d %s"%( cnt, i))
 cnt = cnt + 1


cnt = 0
for i in range(10):
 print ("%d %s"%( i, symbol_list[i]))


maxxx = 0
print ("maxxx: %d %s"%( maxxx, symbol_list[maxxx]))


def findindex ( lib, keys ):
  idex = []
  #library = lib.tolist()
  library = lib
  for k in keys:
    #print(k)
    idex.append(library.index(k))
  #print(idex)
  return(idex)


dx = findindex( symbol_list,  ["1_1_1", "/_1_1"])
print( "1_1_1 index is %s"%(dx))

