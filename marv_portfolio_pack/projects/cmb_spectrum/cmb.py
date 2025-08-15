import numpy as np, matplotlib.pyplot as plt
def sachs_wolfe_cl(ell, As=2.1e-9, ns=0.965): return As*np.power(np.clip(ell,1,None), ns-3.0)
def plot(lmax=1500, As=2.1e-9, ns=0.965, out='cmb.png'):
    ell = np.arange(2, lmax+1); cl = sachs_wolfe_cl(ell, As, ns); Dl = ell*(ell+1)*cl/(2*np.pi)
    plt.figure(figsize=(7,4)); plt.loglog(ell, Dl); plt.tight_layout(); plt.savefig(out, dpi=200); print('Saved', out)
if __name__=='__main__': plot()
