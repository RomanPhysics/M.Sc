#Roman Sultanov
#Imports
from AmpLambda import *
#from AmpXi import *

import csv
with open(r'MCData.csv', 'r', encoding='utf-8') as csvFile:
    reader = csv.reader(csvFile)
    m2pKdat = []
    m2Kπdat = []
    cosθpdat = []
    φpdat = []
    χdat = []
    
    for row in reader:
        m2pKdat.append(float(row[0]))
        m2Kπdat.append(float(row[1]))
        cosθpdat.append(float(row[2]))
        φpdat.append(float(row[3]))
        χdat.append(float(row[4]))
    
    m2pKdat=np.array(m2pKdat)
    m2Kπdat=np.array(m2Kπdat)
    cosθpdat=np.array(cosθpdat)
    φpdat=np.array(φpdat)
    χdat=np.array(χdat)

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

x = m2pKdat
y = m2Kπdat
xbins = 200
ybins = 200
counts, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins])
masked_counts = np.ma.masked_where(counts == 0, counts)
plt.figure(figsize=(8, 6))
plt.imshow(masked_counts.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis', origin='lower', aspect='auto')
cbar=plt.colorbar(pad=0.0)#label='Arbitrary Units')
cbar.set_ticks([])
cbar.set_ticklabels([])
plt.xlabel('$m^2(pK^-)$ [GeV$^2$]')
plt.ylabel('$m^2(K^-\pi^+)$ [GeV$^2$]')

plt.xlim((mp + mK) ** 2-0.1, (mΛ - mπ) ** 2+0.1)  # Adjust x-axis limits
plt.ylim((mK + mπ) ** 2-0.1, (mΛ - mp) ** 2+0.1)  # Adjust y-axis limits
plt.xticks(np.arange(round((mp + mK) ** 2/0.5)*0.5, round((mΛ - mπ) ** 2/0.5)*0.5+0.5,0.5))
plt.yticks(np.arange(round((mK + mπ) ** 2/0.5)*0.5, round((mΛ - mp) ** 2/0.5)*0.5+0.5,0.5))
plt.gca().xaxis.set_ticks_position('both')  # Show ticks on both top and bottom
plt.gca().yaxis.set_ticks_position('both')  # Show ticks on both left and right
import matplotlib.ticker as ticker
plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.tick_params(axis='both', direction='in', length=6, which='both')
plt.tick_params(axis='both', direction='in', length=4, which='minor')
plt.show()

m2pπdat=mΛ**2+mp**2+mK**2+mπ**2 - np.array(m2pKdat) - np.array(m2Kπdat)

fig, axs = plt.subplots(2, 3, figsize=(16, 10))  # 2 rows, 3 columns
axs = axs.flatten()

axs[0].hist(m2pKdat, bins=300, color='red', alpha=0.7, histtype='step', linewidth=1.5)
axs[0].set_xlabel('$m^2(pK^-)$ [GeV$^2$]')
axs[0].set_ylabel('Events')
axs[0].set_xticks(np.arange(round(np.min(m2pKdat)/0.5)*0.5, round(np.max(m2pKdat)/0.5)*0.5+0.5,0.5))
axs[0].set_xlim(1.9, 4.8)
axs[0].grid(False)
axs[0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[0].tick_params(axis='both', direction='in', length=6, which='both')
axs[0].tick_params(axis='both', direction='in', length=4, which='minor')

axs[1].hist(m2pπdat, bins=300, color='red', alpha=0.7, histtype='step', linewidth=1.5)
axs[1].set_xlabel('$m^2(p\pi^+)$ [GeV$^2$]')
axs[1].set_ylabel('Events')
axs[1].set_xticks(np.arange(round(np.min(m2pπdat)/0.5)*0.5, round(np.max(m2pπdat)/0.5)*0.5+0.5,0.5))
axs[1].set_xlim(1.1, 3.4)
axs[1].grid(False)
axs[1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[1].yaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[1].tick_params(axis='both', direction='in', length=6, which='both')
axs[1].tick_params(axis='both', direction='in', length=4, which='minor')

axs[2].hist(m2Kπdat, bins=300, color='red', alpha=0.7, histtype='step', linewidth=1.5)
axs[2].set_xlabel('$m^2(K^-\pi^+)$ [GeV$^2$]')
axs[2].set_ylabel('Events')
axs[2].set_xticks(np.arange(round(np.min(m2Kπdat)/0.5)*0.5, round(np.max(m2Kπdat)/0.5)*0.5+0.5,0.5))
axs[2].set_xlim(0.35, 1.84)
axs[2].grid(False)
axs[2].xaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[2].yaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[2].tick_params(axis='both', direction='in', length=6, which='both')
axs[2].tick_params(axis='both', direction='in', length=4, which='minor')

axs[3].hist(cosθpdat, bins=300, color='red', alpha=0.7, histtype='step', linewidth=1.5)
axs[3].set_xlabel('$\cos(θ_p)$')
axs[3].set_ylabel('Events')
axs[3].set_xticks(np.arange(-1, 1.5, 0.5))
axs[3].set_xlim(-1, 1)
axs[3].grid(False)
axs[3].xaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[3].yaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[3].tick_params(axis='both', direction='in', length=6, which='both')
axs[3].tick_params(axis='both', direction='in', length=4, which='minor')

axs[4].hist(φpdat, bins=300, color='red', alpha=0.7, histtype='step', linewidth=1.5)
axs[4].set_xlabel('$φ_p$')
axs[4].set_ylabel('Events')
axs[4].set_xticks(np.arange(-3, 4, 1))
axs[4].set_xlim(-np.pi, np.pi)
axs[4].grid(False)
axs[4].xaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[4].yaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[4].tick_params(axis='both', direction='in', length=6, which='both')
axs[4].tick_params(axis='both', direction='in', length=4, which='minor')

axs[5].hist(χdat, bins=300, color='red', alpha=0.7, histtype='step', linewidth=1.5)
axs[5].set_xlabel('$χ$')
axs[5].set_ylabel('Events')
axs[5].set_xticks(np.arange(-3, 4, 1))
axs[5].set_xlim(-np.pi, np.pi)
axs[5].grid(False)
axs[5].xaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[5].yaxis.set_minor_locator(ticker.AutoMinorLocator())
axs[5].tick_params(axis='both', direction='in', length=6, which='both')
axs[5].tick_params(axis='both', direction='in', length=4, which='minor')
plt.tight_layout()
plt.show()