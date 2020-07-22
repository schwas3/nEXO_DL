import pickle
import ROOT
import array

h1 = ROOT.TH1F('h1', 'h1', 100, 0, 1)
h2 = ROOT.TH1F('h2', 'h2', 100, 0, 1)
output = pickle.load( open( "./train_float16/save_sens_23.p", "rb" ) , encoding='latin1')
score = output[4][0]
for item in score:
    score, target = float(item[0]), int(item[1])
    if target == 0:
        h1.Fill(score)
    else:
        h2.Fill(score)

c1 = ROOT.TCanvas('c1', 'c1', 800, 600)
c1.SetLogy(1)
h1.SetLineColor(ROOT.kRed)
h2.SetLineColor(ROOT.kBlue)
h1.SetMaximum(1.1*max(h1.GetMaximum(), h2.GetMaximum()))
h1.Draw()
h2.Draw('histsame')
leg1 = ROOT.TLegend(0.4, 0.6, 0.65, 0.85)
leg1.AddEntry(h1, 'Background','l')
leg1.AddEntry(h2, 'Signal', 'l')
ROOT.gStyle.SetOptTitle(0)
ROOT.gStyle.SetOptStat(0)
h1.SetXTitle('DNN output')
h1.SetYTitle('# of events')
leg1.Draw()
leg1.SetBorderSize(0)
c1.Print('dnn_output_sens_1.pdf')
bkg = array.array('d')
sig = array.array('d')
for i in range(h1.GetNbinsX()-3):
    bkg.append(h1.Integral(i,101)*1.0/h1.Integral())
    sig.append(h2.Integral(i,101)*1.0/h2.Integral())
bkg.append(0)
sig.append(0)

for bkgi, sigi in zip(bkg, sig):
    print(bkgi, sigi)
roc = ROOT.TGraph(len(bkg), bkg, sig)
c2 = ROOT.TCanvas('c2', 'c2', 800,600)
cf = c2.DrawFrame(0, 0., 0.1, 1)
cf.SetXTitle('bkg misID')
cf.SetYTitle('signal efficiency')
roc.Draw('c')
roc.SetLineColor(ROOT.kRed)
leg2 = ROOT.TLegend(0.5, 0.2, 0.75, 0.4)
#leg2.AddEntry(roc, 'DNN', 'l')
#leg2.Draw()
c2.Print('dnn_roc_sens_1.pdf')
