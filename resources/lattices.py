import math
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
try: import pandas as pd
except ImportError: pd = None


class Geometry:
    def __init__(self, LAT, l, nnx, rD=0.2, t=None, nny=None):
        self.LAT = LAT
        self.l = l
        self.nnx = nnx
        self.rD = rD
        self.t = t
        self.nny = nny
        if t is None:
            self.rDthickness(rD=rD)
        else:
            self.rDthickness(t=t)

        if LAT.lower() == 'fcc':
            L = float(l * nnx)
            Lmin = L
            if nny is None:
                H0 = 0.96 * L
                Hs = [l * i for i in range(100)]
                H = min(Hs, key=lambda x: abs(x - H0))
                nny = H / l
            else:
                H = float(nny * l)

            if round(nny) % 2.0 == 0.0:
                if H / L >= 0.96:
                    H = H - l
                    nny = H / l
                elif H / L < 0.96:
                    H = H + l
                    nny = H / l

            W = L / 1.25
            a = [L / nnx * i for i in range(nnx + 1)]
            a0 = min(a, key=lambda x: abs(x - (0.75 * W)))
            ai = [a0 + ((l / 2) * (i)) for i in range(nnx)]
            vol = L * H

            nny = int(round(nny))
            totalNodes = int(round((nnx + 1) * (nny + 1) + nnx * nny))
            totalBracketNodes = int(round((nnx + 5) * 3 * 2 + (nnx + 4) * 3 * 2))
            deltaNM = 0.5 * np.sqrt(l * l + l * l)
            iso = False

        elif LAT.lower() == 'square':
            L = float(l * nnx)
            Lmin = L
            if nny is None:
                H0 = 0.96 * L
                Hs = [l * i for i in range(100)]
                H = min(Hs, key=lambda x: abs(x - H0))
                nny = H / l
            else:
                H = float(nny * l)

            if round(nny) % 2.0 == 0.0:
                if H / L >= 0.96:
                    H = H - l
                    nny = H / l
                elif H / L < 0.96:
                    H = H + l
                    nny = H / l

            W = L / 1.25
            a = [L / nnx * i for i in range(nnx + 1)]
            a0 = min(a, key=lambda x: abs(x - (0.75 * W)))
            ai = [a0 + ((l / 2) * (i)) for i in range(nnx)]
            vol = L * H

            nny = int(round(nny))
            totalNodes = int(round((nnx + 1) * (nny + 1)))
            totalBracketNodes = int(round((nnx + 5) * 3 * 2))
            deltaNM = l
            iso = False

        elif LAT.lower() == '45square':
            L = float(2**(1/2) * l * nnx)
            Lmin = L
            if nny is None:
                H0 = 0.96 * L
                Hs = [2**(1/2) * l * i for i in range(100)]
                H = min(Hs, key=lambda x: abs(x - H0))
                nny = H / ((2**(1/2))*l)
            else:
                H = float(nny * (2**(1/2)) * l)

            if round(nny) % 2.0 == 0.0:
                if H / L >= 0.96:
                    H = H - ((2**(1/2))*l)
                    nny = H / ((2**(1/2))*l)
                elif H / L < 0.96:
                    H = H + ((2**(1/2))*l)
                    nny = H / ((2**(1/2))*l)

            W = L / 1.25
            a = [L / nnx * i for i in range(nnx + 1)]
            a0 = min(a, key=lambda x: abs(x - (0.75 * W)))
            ai = [a0 + (((2**(1/2))*l) * (i)) for i in range(nnx)]
            vol = L * H

            nny = int(round(nny))
            totalNodes = int(round((nnx + 1) * (nny + 1) + nnx * nny))
            totalBracketNodes = int(round((nnx + 5) * 3 * 2 + (nnx + 4) * 3 * 2))
            deltaNM = l
            iso = False

        elif LAT.lower() == 'tri':
            if nnx % 2.0 == 1.0:
                nnx = nnx - 1
            L = 0.5 * (3.0 ** 0.5) * l * nnx
            Lmin = L
            if nny is None:
                H0 = 0.96 * L
                Hs = [l * i for i in range(100)]
                H = min(Hs, key=lambda x: abs(x - H0))
                nny = H / l
            else:
                H = float(nny * l)

            if round(nny) % 2.0 == 0.0:
                if H / L >= 0.96:
                    H = H - l
                    nny = H / l
                elif H / L < 0.96:
                    H = H + l
                    nny = H / l

            W = L / 1.25
            a = [L / (nnx / 2) * i for i in range(nnx + 1)]
            a0 = min(a, key=lambda x: abs(x - (0.75 * W)))
            ai = [a0 + ((0.5 * (3.0 ** 0.5) * l) * (i)) for i in range(nnx)]
            vol = L * H

            nny = int(round(nny))
            totalNodes = int(round(((nnx / 1.99999) + 1) * (nny + 1)) +
                             round((nnx / 1.99999) * nny))
            totalBracketNodes = int(round(((nnx / 1.99999) + 3) * 3 * 2) +
                                    round(((nnx / 1.99999) + 2) * 3 * 2) +
                                    2 * (nnx / 2.0 + 2))
            deltaNM = l
            iso = True

        elif LAT.lower() == 'kagome':
            L = l * (2.0 * nnx - 1)
            Lmin = L - 3*l
            if nny is None:
                H0 = 0.96 * L
                Hs = [(3.0 ** 0.5) * l * i for i in range(100)]
                H = min(Hs, key=lambda x: abs(x - H0))
                nny = H / ((3.0 ** 0.5) * l)
            else:
                H = float(nny * (3.0 ** 0.5) * l)

            if round(nny) % 2.0 == 0.0:
                if H / L >= 0.96:
                    H = H - ((3.0 ** 0.5) * l)
                    nny = H / ((3.0 ** 0.5) * l)
                elif H / L < 0.96:
                    H = H + ((3.0 ** 0.5) * l)
                    nny = H / ((3.0 ** 0.5) * l)

            W = L / 1.25
            if round(nny) % 4 == 3:
                a = [2 * L * i / (2 * nnx - 1) + 0.5 * l for i in range(nnx + 1)]
            elif round(nny) % 4 == 1:
                a = [2 * L * i / (2 * nnx - 1) + 1.5 * l for i in range(nnx + 1)]
            else:
                a = [2 * L * i / (2 * nnx - 1) + 0.5 * l for i in range(nnx + 1)]

            a0 = min(a, key=lambda x: abs(x - (0.75 * W)))
            ai = [a0 + ((2 * l) * (i)) for i in range(nnx)]
            vol = L * H

            nny = int(round(nny))
            totalNodes = int(round((2 * nnx * (nny + 1)) +
                                   (nnx - 1) * math.ceil(nny / 2.0) +
                                   (nnx) * math.floor(nny / 2)))
            totalBracketNodes = int(round(((2 * nnx + 4) * 3 + (nnx + 2) * 2 + (nnx + 1)) * 2))
            deltaNM = l
            iso = True

        elif LAT.lower() == 'hex':
            L = (3.0 ** 0.5) * l * nnx
            Lmin = L - ((3.0 ** 0.5) * l)
            if nny is None:
                H0 = 0.96 * L
                Hs = [(0.5 * l) + (1.5 * l * i) for i in range(100)]
                H = min(Hs, key=lambda x: abs(x - H0))
                nny = (H - (0.5 * l)) / (1.5 * l)
            else:
                H = float((0.5 * l) + (1.5 * l * nny))

            if round(nny) % 2.0 == 0.0:
                if H / L >= 0.96:
                    H = H - 1.5 * l
                    nny = (H - (0.5 * l)) / (1.5 * l)
                elif H / L < 0.96:
                    H = H + 1.5 * l
                    nny = (H - (0.5 * l)) / (1.5 * l)

            nny = int(round(nny))
            W = L / 1.25

            if round(nny) % 4 == 3:
                a = [((3.0 ** 0.5) / 2) * l + (L - ((3.0 ** 0.5) * l)) / (nnx - 1) * i
                     for i in range(nnx + 1)]
            elif round(nny) % 4 == 1:
                a = [L / nnx * i for i in range(nnx + 1)]
            else:
                a = [L / nnx * i for i in range(nnx + 1)]

            a0 = min(a, key=lambda x: abs(x - (0.75 * W)))
            ai = [a0 + (((3.0 ** 0.5) * (l / 2)) * (i)) for i in range(nnx)]
            vol = L * H

            totalNodes = int(round(2 * (nnx) * math.ceil(nny / 2.0) +
                                   2 * (nnx + 1) * math.ceil(nny / 2.0)))
            totalBracketNodes = int(round(((nnx + 5) * 4 + (nnx + 4) * 4) * 2 + 4))
            deltaNM = l
            iso = True

        self.nnx = nnx
        self.nny = nny
        self.L = L
        self.Lmin = Lmin
        self.H = H
        self.W = W
        self.a0 = a0
        self.ai = ai
        self.vol = vol
        self.totalNodes = totalNodes
        self.totalBracketNodes = totalBracketNodes
        self.deltaNM = deltaNM
        self.B = 0.5 * self.W
        self.iso = iso
        self.ftcalc_done = False

    def rDthickness(self, t=None, rD=None):
        if self.LAT.lower() == "fcc":
            A = 2*(1+np.sqrt(2))
        elif "square" in self.LAT.lower():
            A = 2
        elif self.LAT.lower() == "tri":
            A = 2*np.sqrt(3)
        elif self.LAT.lower() == "kagome":
            A = np.sqrt(3)
        elif self.LAT.lower() == "hex":
            A = 2/np.sqrt(3)
            
        if t is not None:
            self.t = t
            self.rD = A*(np.asarray(t)/self.l)
        elif rD is not None:
            self.rD = rD
            self.t = (self.l*np.asarray(rD))/A

    def stiffnessMatrix(self, stiffCalc=False):
        nnx = 10
        if self.LAT.lower() == "fcc":
            nny = nnx
            L = float(self.l * nnx)
            H = float(self.l * nny)
            vol = L * H
            if stiffCalc:
                if stiffCalc.lower() == "unit":
                    vol = self.l ** 2
                elif stiffCalc.lower() == "lattice":
                    vol = L * H
        elif self.LAT.lower() == "tri":
            if stiffCalc:
                nny = nnx
                L = (3 ** 0.5) * self.l * nnx
                H = self.l * nny
                vol = L * H
                if stiffCalc.lower() == "unit":
                    vol = self.l * (2 * self.l * (3 ** 0.5) / 2)
                elif stiffCalc.lower() == "lattice":
                    vol = L * H
            else:
                nnx = nnx * 2
                nny = nnx / 2
                L = 0.5 * (3.0 ** 0.5) * self.l * nnx
                H = self.l * nny
                vol = L * H
        elif self.LAT.lower() == "kagome":
            nny = nnx
            L = self.l * (2.0 * nnx - 1)
            H = (3.0 ** 0.5) * self.l * nny
            vol = L * H
            if stiffCalc:
                if stiffCalc.lower() == "unit":
                    vol = (3 * self.l) * (4 * self.l * ((3 ** 0.5) / 2))
                elif stiffCalc.lower() == "lattice":
                    vol = L * H
        elif self.LAT.lower() == "hex":
            L = (3 ** 0.5) * self.l * nnx
            if stiffCalc:
                nny = nnx
                H = 3 * self.l * nny
                vol = L * H
                if stiffCalc.lower() == "unit":
                    vol = (3 * self.l) * (2 * self.l * ((3 ** 0.5) / 2))
                elif stiffCalc.lower() == "lattice":
                    vol = L * H
            else:
                nny = nnx * 2 + 1
                H = 3 * self.l * nny
                vol = L * H
        self.nnx = nnx
        self.nny = int(round(nny))
        self.L = L
        self.H = H
        self.vol = vol

    def FTcalc(self):
        if self.ftcalc_done:
            return

        self.a0 = self.a0 - 0.25 * self.W
        if self.LAT.lower() == "fcc":
            self.ai = [self.a0 + ((self.l / 2) * (i)) for i in range(self.nnx)]
        elif "square" in self.LAT.lower():
            self.ai = [self.a0 + ((self.l * 2**(1/2)) * (i)) for i in range(self.nnx)]
        elif self.LAT.lower() == "tri":
            self.ai = [self.a0 + ((0.5 * (3.0 ** 0.5) * self.l) * (i)) for i in range(self.nnx)]
        elif self.LAT.lower() == "kagome":
            self.ai = [self.a0 + ((2 * self.l) * (i)) for i in range(self.nnx)]
        elif self.LAT.lower() == "hex":
            self.ai = [self.a0 + (((3.0 ** 0.5) * (self.l / 2)) * (i)) for i in range(self.nnx)]
        self.ftcalc_done = True

    def nodeCount(self, mode=False, stiffMatrix=False):
        if self.LAT.lower() == "fcc" or self.LAT.lower() == "45square":
            totalNodes = int(round((self.nnx + 1) * (self.nny + 1) + self.nnx * self.nny))
            totalBracketNodes = int(round((self.nnx + 5) * 3 * 2 + (self.nnx + 4) * 3 * 2))
            if mode and mode.lower() == "fracture":
                self.totalNodes = totalNodes - round(self.nnx / 1.66666667)
            elif mode and mode.lower() == "ductile":
                self.totalNodes = totalNodes + totalBracketNodes - 8
            if stiffMatrix:
                nnx, nny = 10, 10
                self.totalNodes = int((nnx + 1) * (nny + 1)) + int(nnx * nny)
        elif self.LAT.lower() == "square":
            totalNodes = int(round((self.nnx + 1) * (self.nny + 1)))
            totalBracketNodes = int(round((self.nnx + 5) * 3 * 2))
            if mode and mode.lower() == "fracture":
                self.totalNodes = totalNodes
            elif mode and mode.lower() == "ductile":
                self.totalNodes = totalNodes + totalBracketNodes - 4
            if stiffMatrix:
                nnx, nny = 10, 10
                self.totalNodes = int((nnx + 1) * (nny + 1))
        elif self.LAT.lower() == "tri":
            totalNodes = int(round(((self.nnx / 1.99999) + 1) * (self.nny + 1)) +
                             round((self.nnx / 1.99999) * self.nny))
            totalBracketNodes = int(round(((self.nnx / 1.99999) + 3) * 3 * 2) +
                                    round(((self.nnx / 1.99999) + 2) * 3 * 2) +
                                    2 * (self.nnx / 2.0 + 2))
            if mode and mode.lower() == "fracture":
                self.totalNodes = totalNodes - round(self.nnx / 3.33333333)
            elif mode and mode.lower() == "ductile":
                self.totalNodes = totalNodes + totalBracketNodes - 4
            if stiffMatrix:
                nnx, nny = 10, 10
                self.totalNodes = int((nnx + 1) * (nny + 1)) + int(nnx * nny)
        elif self.LAT.lower() == "kagome":
            totalNodes = int(round((2 * self.nnx * (self.nny + 1)) +
                                   (self.nnx - 1) * math.ceil(self.nny / 2.0) +
                                   (self.nnx) * math.floor(self.nny / 2)))
            totalBracketNodes = int(round(((2 * self.nnx + 4) * 3 + (self.nnx + 2) * 2 + (self.nnx + 1)) * 2))
            if mode and mode.lower() == "fracture":
                self.totalNodes = totalNodes - round(self.nnx / 1.75)
            elif mode and mode.lower() == "ductile":
                self.totalNodes = totalNodes + totalBracketNodes - 16
            if stiffMatrix:
                nnx, nny = 10, 10
                self.totalNodes = int((2*nnx*(nny+1)) + (nnx-1)*math.ceil(nny/2.0) + (nnx)*math.floor(nny/2))
        elif self.LAT.lower() == "hex":
            totalNodes = int(round(2 * (self.nnx) * math.ceil(self.nny / 2.0) +
                                   2 * (self.nnx + 1) * math.ceil(self.nny / 2.0)))
            totalBracketNodes = int(round(((self.nnx + 5) * 4 + (self.nnx + 4) * 4) * 2 + 4))
            if mode and mode.lower() == "fracture":
                self.totalNodes = totalNodes
            elif mode and mode.lower() == "ductile":
                self.totalNodes = totalNodes + totalBracketNodes - 12
            if stiffMatrix:
                nnx, nny = 10, 10
                self.totalNodes = ((2*nny) * (nnx+1)) + (((2*nny)+1) * nnx) + 50

    def UTval(self):
        self.nnx = 20
        self.l = 10.0
        self.nny = 18
        self.H = self.nny * self.l
        self.vol = self.L * self.H
        self.totalNodes = int(round(((self.nnx / 1.99999) + 1) * (self.nny + 1)) +
                              round((self.nnx / 1.99999) * self.nny))
        self.totalBracketNodes = int(round(((self.nnx / 1.99999) + 3) * 3 * 2) +
                                     round(((self.nnx / 1.99999) + 2) * 3 * 2) +
                                     2 * (self.nnx / 2.0 + 2))

    def brackets(self): # DELETE? add vol if don't delete
        nnx = self.nnx + 4
        nny = self.nny + 6
        H = self.l * nny
        if self.LAT.lower() == "fcc" or self.LAT.lower() == "fcc2":
            L = float(self.l * nnx)
            totalNodes = int(round((nnx + 1) * (nny + 1) + nnx * nny))
            totalBracketNodes = int(round((nnx + 5) * 3 * 2 + (nnx + 4) * 3 * 2))
        elif self.LAT.lower() == "tri":
            L = 0.5 * (3.0 ** 0.5) * self.l * nnx         
            totalNodes = int(round(((nnx / 1.99999) + 1) * (nny + 1)) + round((nnx / 1.99999) * nny))
            totalBracketNodes = int(round(((nnx / 1.99999) + 3) * 3 * 2) +
                                    round(((nnx / 1.99999) + 2) * 3 * 2) +
                                    2 * (nnx / 2.0 + 2))
        elif self.LAT.lower() == "kagome":
            nnx = self.nnx + 2
            L = self.l * (2.0 * nnx - 1)
            H = (3.0 ** 0.5) * self.l * nny
            totalNodes = int(round((2*nnx*(nny+1)) + (nnx-1)*math.ceil(nny/2.0) + (nnx)*math.floor(nny/2)))
            totalBracketNodes = int(round(((2 * nnx + 4) * 3 + (nnx + 2) * 2 + (nnx + 1)) * 2))
        elif self.LAT.lower() == "hex":
            nny = self.nny + 8
            L = (3.0 ** 0.5) * self.l * nnx
            H = (0.5 * self.l) + (1.5 * self.l * nny)
            totalNodes = int(round(2*(nnx)*math.ceil(nny/2.0) + 2*(nnx+1)*math.ceil(nny/2.0)))
            totalBracketNodes = int(round(((nnx + 5) * 4 + (nnx + 4) * 4) * 2 + 4))
        self.nnx = nnx
        self.nny = int(round(nny))
        self.L = L
        self.H = H
        self.totalNodes = totalNodes
        self.totalBracketNodes = totalBracketNodes
        self.B = 0.5 * self.W


def pStrainProperties(E, v, v_s=None, B=None, b=None, rD=None, typ="bulk"):
    if typ.lower() == "bulk":
        return E/(1-v**2), v/(1-v)
    elif typ.lower() == "lattice":
        return E/(1 - (B*(rD**(b-1))*(v_s**2))), (v + (B*(rD**(b-1))*(v_s**2)))/(1 - (B*(rD**(b-1))*(v_s**2)))

def prismatic_plane_strain(C, rD=None, E_s=123e9, v_s=0.3,
                          C_state="plane_stress", crack_axis=0):
    """Reduce a prismatic lattice and obtain its elastic mode-I modulus.

    C uses engineering shear and Voigt order (1, 2, 6) in specimen (x, y).
    For an unconstrained tensor, the rule of mixtures supplies E_3=rD*E_s
    and nu_31=nu_32=v_s (nu_ij is strain in j under stress in i). The
    returned S_1236 is the normal/in-plane-shear compliance block, not a
    complete 6-by-6 tensor: out-of-plane shear moduli are not assumed.

    C_state='plane_strain' accepts an already-reduced tensor, including
    homogenisation with plane-strain-equivalent beam inputs. It bypasses
    the mixture/reduction, preventing a second plane-strain correction.

    crack_axis=0 means crack along x and opening load along y. S_pe is
    ordered (crack, opening, engineering shear); Bao's lambda=S11/S22
    and M=(1/S11)*sqrt(2*lambda**1.5/(1+chi)) use those axes. This does
    not establish the axis mapping of an orthotropic C(T) calibration.
    Every modulus must use the same stress unit as C and E_s.
    """
    C = np.asarray(C, dtype=float)
    if (C.shape != (3, 3) or not np.isfinite(C).all()
            or not np.allclose(C, C.T, rtol=1e-8, atol=0)):
        raise ValueError("C must be a finite symmetric 3-by-3 engineering-shear tensor.")
    if np.linalg.eigvalsh(C).min() <= 0:
        raise ValueError("C must be positive definite.")
    if crack_axis not in (0, 1):
        raise ValueError("crack_axis must be 0 (x) or 1 (y).")
    S_input = np.linalg.inv(C)
    S_full = None
    if C_state == "plane_stress":
        if (rD is None or not np.isfinite(rD) or not 0 < rD <= 1
                or not np.isfinite(E_s) or E_s <= 0
                or not np.isfinite(v_s) or not -1 < v_s < .5):
            raise ValueError("The mixture model requires 0<rD<=1, E_s>0 and -1<v_s<0.5.")
        E3 = rD * E_s
        S_full = np.zeros((4, 4))
        inplane = [0, 1, 3]
        S_full[np.ix_(inplane, inplane)] = S_input
        S_full[2, 2] = 1 / E3
        S_full[0, 2] = S_full[2, 0] = -v_s / E3
        S_full[1, 2] = S_full[2, 1] = -v_s / E3
        if np.linalg.eigvalsh(S_full).min() <= 0:
            raise ValueError("The prismatic mixture compliance is not positive definite; check finite-density constants.")
        coupling = S_full[inplane, 2]
        S_xy_pe = S_input - np.outer(coupling, coupling) / S_full[2, 2]
    elif C_state == "plane_strain":
        E3 = None
        S_xy_pe = S_input.copy()
    else:
        raise ValueError("C_state must be 'plane_stress' or 'plane_strain'.")
    # Bao's aligned orthotropic formula excludes normal/shear coupling.
    scale = np.max(np.abs(S_xy_pe))
    if np.max(np.abs(S_xy_pe[:2, 2])) > 1e-8 * scale:
        raise ValueError("Aligned orthotropic mode-I conversion requires zero S16 and S26.")
    order = [crack_axis, 1-crack_axis, 2]
    S_pe = S_xy_pe[np.ix_(order, order)]
    lambda_pe, chi_pe = calc_anisoParams(S=S_pe)
    M_I_pe = np.sqrt(2 * lambda_pe**1.5 / (1 + chi_pe)) / S_pe[0, 0]
    return {"S_input": S_input, "S_1236": S_full, "S_xy_pe": S_xy_pe,
            "S_pe": S_pe, "E3": E3, "lambda_pe": lambda_pe,
            "chi_pe": chi_pe, "M_I_pe": M_I_pe,
            "E_pe": 1 / S_pe[0, 0], "nu_pe": -S_pe[0, 1] / S_pe[0, 0]}


def effProperties(LAT, geom, E_s=123e9, v_s=0.3, mode="stiff", C=None,
                  C_state="plane_stress"):
    """Return input-state E, nu and plane-strain fracture M, nu.

    For FCC, E is a directional Young's modulus, not the fracture modulus.
    Physical-material C is mixture-reduced once; C_state='plane_strain'
    instead declares that the supplied C already includes the constraint.
    Existing non-FCC scaling/affine approximations are retained; these are
    not a common finite-density homogenisation or a revised dataset.
    """
    if LAT.lower() == "fcc":
        if mode != "stiff":
            raise ValueError("FCC requires a stiffness tensor; no isotropic FCC scaling law is defined.")
        if C is None:
            if C_state != "plane_stress":
                raise ValueError("Supply C explicitly when it is already plane-strain reduced.")
            C = calcC_mohr(copy.deepcopy(geom), "unit", E_s)[0]
        elastic = prismatic_plane_strain(C, geom.rD, E_s, v_s, C_state=C_state)
        S = elastic["S_input"]
        return 1/S[0, 0], -S[0, 1]/S[0, 0], elastic["M_I_pe"], elastic["nu_pe"]
    if C_state != "plane_stress":
        raise ValueError("For already-reduced non-FCC tensors, use prismatic_plane_strain directly.")
    if LAT.lower() == "square":
        B, b = 1/2, 1
    elif LAT.lower() == "45square":
        B, b = 1/4, 3
    elif LAT.lower() == "tri":
        B, b = 1/3, 1
    elif LAT.lower() == "kagome":
        B, b = 1/3, 1
    elif LAT.lower() == "hex":
        B, b = 3/2, 3
    if mode == "stiff":
        if C is None:
            C = calcC_mohr(copy.deepcopy(geom), "unit", E_s)[0]
        E, v, _ = calc_IsoEffProperties(C)
        if not geom.iso:
            lambda_aniso, chi_aniso = calc_anisoParams(C=C)
            E = (C[0,0]*(1-v**2)) * np.sqrt((2*(lambda_aniso**(3/2)))/(1+chi_aniso))
    else:
        E = E_s * B * (geom.rD ** b)
        if LAT.lower() == "square":
            v = 0
        elif LAT.lower() == "45square":
            v = 1
        elif LAT.lower() == "tri":
            v = 1/3
        elif LAT.lower() == "kagome":
            v = 1/3
        elif LAT.lower() == "hex":
            v = 1
    E_pe, v_pe = pStrainProperties(E, v, v_s=v_s, B=B, b=b, rD=geom.rD, typ="lattice")
    return E, v, E_pe, v_pe


def find_nodes(LAT, geom, dis, mode='lattice', stiff=False, path=None, sim=1):
    l = geom.l
    if mode.lower() == "unit":
        if LAT.lower() == "fcc":
            nodes = np.array([[0,0,0],
                              [0,l,0],
                              [l,l,0],
                              [l,0,0],
                              [l/2,l/2,0]])
        elif LAT.lower() == "tri":
            nodes = np.array([[0,0,0],
                              [l,0,0],
                              [l/2,l*(np.sqrt(3)/2),0],
                              [0,l*(np.sqrt(3)/2),0],
                              [l,l*(np.sqrt(3)/2),0],
                              [0,2*l*(np.sqrt(3)/2),0],
                              [l,2*l*(np.sqrt(3)/2),0]])
        elif LAT.lower() == "kagome":
            nodes = np.array([[l*(np.sqrt(3)/2),0,0],
                              [l*(np.sqrt(3)/2),l,0],
                              [0,1.5*l,0],
                              [l*(np.sqrt(3)/2),2*l,0],
                              [l*(np.sqrt(3)/2),3*l,0],
                              [2*l*(np.sqrt(3)/2),2.5*l,0],
                              [3*l*(np.sqrt(3)/2),3*l,0],
                              [3*l*(np.sqrt(3)/2),2*l,0],
                              [4*l*(np.sqrt(3)/2),1.5*l,0],
                              [3*l*(np.sqrt(3)/2),l,0],
                              [3*l*(np.sqrt(3)/2),0,0],
                              [2*l*(np.sqrt(3)/2),0.5*l,0]])
        elif LAT.lower() == "hex":
            nodes = np.array([[0,l*(np.sqrt(3)/2),0],
                              [0.5*l,l*(np.sqrt(3)/2),0],
                              [l,2*l*(np.sqrt(3)/2),0],
                              [2*l,2*l*(np.sqrt(3)/2),0],
                              [2.5*l,l*(np.sqrt(3)/2),0],
                              [3*l,l*(np.sqrt(3)/2),0],
                              [2*l,0,0],
                              [l,0,0]])
        Dnodes = nodes

    elif mode.lower() == "lattice":
        pDir = os.fspath(path) if path is not None else (
            "Z:/p1/sims/Ti/stiffMatrix" if stiff else "Z:/p1/sims/Ti")
        nodeFile = pDir + "/transfer/IN-nDuctile-" + LAT + "-" + str(geom.nnx) + "-per-1.csv"
        
        nodes_df = pd.read_csv(nodeFile, header=None, usecols=[1, 2])
        nodes = nodes_df.to_numpy()
    
        if stiff:
            if LAT.lower() == "tri":
                ys = [0, geom.H]
                xs = [l*(np.sqrt(3)/2) + l*np.sqrt(3)*i for i in range(geom.nnx)]
                add_nodes = []
                for y in ys:
                    for x in xs:
                        add_nodes.append([x, y])
                nodes = np.append(nodes, add_nodes, axis=0)

            del_nodes1 = np.argwhere(nodes[:, 1] > geom.H+1e-6)
            del_nodes2 = np.argwhere(nodes[:, 1] < 0)
            del_nodes3 = np.argwhere(nodes[:, 0] > geom.L+1e-6)
            del_nodes4 = np.argwhere(nodes[:, 0] < 0)
            del_nodes = np.concatenate([del_nodes1, del_nodes2, del_nodes3, del_nodes4])
            nodes = np.delete(nodes, del_nodes, axis=0)
        Dnodes = nodes
      
        if "disnodes" in dis.lower():
            DnodeFile = pDir + "/transfer/IN-nDuctile-" + LAT + "-" + str(geom.nnx) + "-20disNodes-lhs-all-" + str(sim) + ".csv"
            Dnodes_df = pd.read_csv(DnodeFile, header=None, usecols=[1, 2])
            Dnodes = Dnodes_df.to_numpy()
            if stiff:
                if LAT.lower() == "tri":
                    Dnodes = np.append(Dnodes, add_nodes, axis=0)
                elif LAT.lower() == "hex":
                    topNodes_idxs = np.argwhere(nodes[:, 1] == geom.H).flatten()
                    DtopNodes_idxs1 = np.argwhere(Dnodes[:, 1] > geom.H-0.25*geom.l)
                    DtopNodes_idxs2 = np.argwhere(Dnodes[:, 1] < geom.H+0.25*geom.l)
                    DtopNodes_idxs = np.intersect1d(DtopNodes_idxs1, DtopNodes_idxs2)
                    Dnodes[DtopNodes_idxs] = nodes[topNodes_idxs]
                Dnodes = np.delete(Dnodes, del_nodes, axis=0)
    return nodes, Dnodes

def insidePoint(bl, tr, p) :
   if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
      return True
   else :
      return False

def connectivity(LAT, nodes, geom, job=None, stiff=False, mode=None):
    radius = geom.l + geom.l*1e-3
    dummyElem = []
    count = 0
    x_coord = nodes.shape[-1]-3 if stiff else nodes.shape[-1]-2
    y_coord = nodes.shape[-1]-2 if stiff else nodes.shape[-1]-1
    for ii in range(len(nodes)):
        # FCC centres lie at half-cell coordinates; this test must not depend
        # on whether the dimensional cell size happens to be an even integer.
        if LAT.lower() == "fcc" and np.all(np.isclose(
                np.mod(2*nodes[ii, [x_coord, y_coord]]/geom.l, 2),
                1., rtol=0, atol=1e-8)):
            continue
        distance = np.sqrt(np.array(nodes[ii, int(x_coord)] - nodes[:, int(x_coord)])**2 + np.array(nodes[ii, int(y_coord)] - nodes[:, int(y_coord)])**2)
        inside = np.argwhere(distance <= radius)
        nearNodes = np.setdiff1d(inside.astype(int), [ii])
        for jj in range(len(nearNodes)):
            dummyElem.append([count,ii,nearNodes[jj]])
            count = count + 1            
    for i in range(len(dummyElem)):
        for j in range(len(dummyElem)):
            if (dummyElem[i][1] == dummyElem[j][2]):
                if (dummyElem[i][2] == dummyElem[j][1]):
                    dummyElem[j][:] = [0, 0, 0]
                    break
    indexRemove = []
    for i in range(0,len(dummyElem)):
        if (dummyElem[i][0] == 0 and dummyElem[i][1] == 0 and dummyElem[i][2] == 0):
            indexRemove.append(i)
        if stiff:
            if LAT.lower() == "tri" and mode.lower() == "unit":
                if (dummyElem[i][1] == 3 and dummyElem[i][2] == 4) or (dummyElem[i][1] == 4 and dummyElem[i][2] == 3):
                    indexRemove.append(i)
                elif (dummyElem[i][1] == 0 and dummyElem[i][2] == 3) or (dummyElem[i][1] == 3 and dummyElem[i][2] == 0):
                    indexRemove.append(i)
                elif (dummyElem[i][1] == 3 and dummyElem[i][2] == 5) or (dummyElem[i][1] == 5 and dummyElem[i][2] == 3):
                    indexRemove.append(i)
                elif (dummyElem[i][1] == 1 and dummyElem[i][2] == 4) or (dummyElem[i][1] == 4 and dummyElem[i][2] == 1):
                    indexRemove.append(i)
                elif (dummyElem[i][1] == 4 and dummyElem[i][2] == 6) or (dummyElem[i][1] == 6 and dummyElem[i][2] == 4):
                    indexRemove.append(i)
            elif LAT.lower() == "tri" and mode.lower() == "lattice":
                if ((nodes[dummyElem[i][1]][1] == 0 and nodes[dummyElem[i][2]][1] == 0) or 
                    (nodes[dummyElem[i][1]][1] == geom.H and nodes[dummyElem[i][2]][1] == geom.H)):
                    indexRemove.append(i)
    realElem = np.delete(dummyElem, [indexRemove], axis=0)
    if stiff:
        realElem = realElem
    else:
        realElem = realElem + 1

    if mode and mode.lower() == "fracture":
        xCrS = [-0.1*geom.W, geom.H/2-geom.l*0.2]                                         # crack starting point bottomLeft
        xCrE = [geom.a0 - 0.2*geom.l, geom.H/2+geom.l*0.2]
    
        delElems = []
        for ik in range(0,len(realElem)):
            offset = 0 if stiff else 1
            x1 = nodes[int(realElem[ik][1])-offset][x_coord]
            x2 = nodes[int(realElem[ik][2])-offset][x_coord]
            y1 = nodes[int(realElem[ik][1])-offset][y_coord]
            y2 = nodes[int(realElem[ik][2])-offset][y_coord]
            midPointX = (x1+x2)/2
            midPointY = (y1+y2)/2
            point = (midPointX,midPointY)
            insideTest = insidePoint(xCrS,xCrE,point)
            if insideTest:
                # Candidate element labels contain gaps after duplicate removal.
                # np.delete consumes current row indices, not those labels.
                delElems.append(ik)

        delElems = np.array(delElems, dtype=int)
        realElem = np.delete(realElem, delElems, 0)
    
    for i in range(len(realElem)):
        realElem[i][0] = i+1

    if job is not None:
        with open(job.split('.inp')[0]+"\\"+"NodesElems.csv", 'a') as f:
            f.write("*Elems\n")
            for elem in realElem:
                f.write("{}, {}, {}\n".format(int(elem[0]), int(elem[1]), int(elem[2])))
    return realElem


def plot_lattice(elems, nodes, geom):
    fig, ax = plt.subplots(1,1)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    thickness = np.broadcast_to(np.asarray(geom.t, dtype=float), (len(elems),))
    for elem, tt in zip(elems, thickness):
        node1 = nodes[elem[1]]
        node2 = nodes[elem[2]]
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], linewidth=tt*2, c='black')
    plt.show()


### Stiffness Matrix Calculation Functions
def edgeElems(nodes, elems, geom):
    t_old = np.broadcast_to(np.asarray(geom.t, dtype=float), (len(elems),))
    t_new = t_old.copy()
    min_x, max_x = min(nodes[:,0]), max(nodes[:,0])
    min_y, max_y = min(nodes[:,1]), max(nodes[:,1])
    for i, elem in enumerate(elems):
        node1 = nodes[elem[1]]
        node2 = nodes[elem[2]]
        if (node1[0] == max_x and node2[0] == max_x) or (node1[0] == min_x and node2[0] == min_x):
            t_new[i] = t_old[i]/2
        elif (node1[1] == max_y and node2[1] == max_y) or (node1[1] == min_y and node2[1] == min_y):
            t_new[i] = t_old[i]/2
    geom.t = t_new
    return geom

def get_n0s(nodes, realElem):
    n0s = []
    for elem in realElem:
        n0 = nodes[elem[2]] - nodes[elem[1]]
        n0s.append(n0)
    n0s = np.array(n0s)
    return n0s

def get_ns(n0s):
    ns = []
    for n0 in n0s:
        ns.append(n_values(n0))
    return np.array(ns)
    
def n_values(n0):
    n0 = np.asarray(n0, dtype=float)
    if n0.ndim != 1 or n0.size < 2 or not np.isfinite(n0[:2]).all() or not np.any(n0[:2] != 0):
        raise ValueError("A strut direction requires a finite nonzero in-plane vector.")
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.array([np.cos(np.arctan([n0[1]/n0[0]]))[0], np.sin(np.arctan([n0[1]/n0[0]]))[0]])

def get_Nmatrix(ns):
    N = []
    for i in range(len(ns)):
        N.append([ns[i][0]*ns[i][0], ns[i][1]*ns[i][1], ns[i][0]*ns[i][1]])
    return np.array(N)

def calc_c(n0, geom, i):
    return (geom.t[i]*(np.sqrt(n0[0]**2 + n0[1]**2))) / (geom.vol)

def calcC_mohr(geom, mode, E_s=123e9, dis='per', count=0, plot=False):
    LAT  = geom.LAT
    geom.stiffnessMatrix(stiffCalc=mode)
    
    nodes, _ = find_nodes(LAT, geom, dis, mode=mode, stiff=True)
    elems = connectivity(LAT, nodes, geom, stiff=True, mode=mode)
    geomC = edgeElems(nodes, elems, copy.deepcopy(geom))
    
    Cs = []
    for nSim in range(count+1):
        if nSim == 0:
            dis_, nSim_ = "per", 1
        else:
            dis_, nSim_ = "20disNodes-lhs-all", nSim
        _, Dnodes = find_nodes(LAT, geom, dis_, mode=mode, stiff=True, sim=nSim_)
        
        if plot:
            plot_lattice(elems, Dnodes, geomC)
            
        n0s = get_n0s(Dnodes, elems)
        ns = get_ns(n0s)
        Nmatrix = get_Nmatrix(ns)

        c0matrix = np.zeros((len(ns),len(ns)))
        for i in range(len(ns)):
            c0matrix[i][i] = calc_c(n0s[i], geomC, i)
        
        Cmatrix = E_s*np.matmul(np.matmul(Nmatrix.T, c0matrix), Nmatrix).round(10)
        Cs.append(Cmatrix)

    return np.array(Cs)


def calcC_sims(LAT, nnx, dis="per", count=0, pDir=r"Z:\p1\sims\Ti\stiffMatrix\Cmatrix\\"):
    from resources.calculations import get_ductileData
    Cs = []
    for nSim in range(count+1):
        if nSim == 0:
            dis_, nSim_ = "per", 1
        else:
            dis_, nSim_ = "20disNodes-lhs-all", nSim
        CSVout11 = f"{pDir}transfer\OUT-Ductile-{LAT}-{int(nnx)}-CmatrixA-{dis_}-{nSim_}-0.csv"
        CSVout12 = f"{pDir}transfer\OUT-Ductile-{LAT}-{int(nnx)}-CmatrixA-{dis_}-{nSim_}-1.csv"
        CSVout22 = f"{pDir}transfer\OUT-Ductile-{LAT}-{int(nnx)}-CmatrixB-{dis_}-{nSim_}-0.csv"
        CSVout21 = f"{pDir}transfer\OUT-Ductile-{LAT}-{int(nnx)}-CmatrixB-{dis_}-{nSim_}-1.csv"
        CSVout33_1 = f"{pDir}transfer\OUT-Ductile-{LAT}-{int(nnx)}-CmatrixC-{dis_}-{nSim_}-0.csv"
        CSVout33_2 = f"{pDir}transfer\OUT-Ductile-{LAT}-{int(nnx)}-CmatrixC-{dis_}-{nSim_}-1.csv"
        CSVout33_3 = f"{pDir}transfer\OUT-Ductile-{LAT}-{int(nnx)}-CmatrixC-{dis_}-{nSim_}-3.csv"
        CSVout33_4 = f"{pDir}transfer\OUT-Ductile-{LAT}-{int(nnx)}-CmatrixC-{dis_}-{nSim_}-4.csv"
        UTdf11 = get_ductileData(CSVout11, crit=0.25)
        UTdf12 = get_ductileData(CSVout12, crit=0.25)
        UTdf22 = get_ductileData(CSVout22, crit=0.25)
        UTdf21 = get_ductileData(CSVout21, crit=0.25)
        UTdf33_1 = get_ductileData(CSVout33_1, crit=0.25)
        UTdf33_2 = get_ductileData(CSVout33_2, crit=0.25)

        C_11 = np.mean(UTdf11.y/UTdf11.x)
        C_12 = np.mean(UTdf12.y/UTdf12.x)
        C_22 = np.mean(UTdf22.y/UTdf22.x)
        C_21 = np.mean(UTdf21.y/UTdf21.x)
        C_33_1 = np.mean(UTdf33_1.y/(2*UTdf33_1.x))
        C_33_2 = np.mean(UTdf33_2.y/(2*UTdf33_2.x))

        C = np.array([[(C_11+C_22)/2, (C_12+C_21)/2, 0],
                      [(C_12+C_21)/2, (C_11+C_22)/2, 0],
                      [0, 0, ((C_33_1+C_33_2)/2)]])
                      
        Cs.append(C)

    return np.array(Cs)


def calc_Compliance(C):
    return np.linalg.inv(C)


def check_isotropy(C, rtol=1e-3, atol=1e-8):
    """Test 2D isotropic stiffness form with engineering shear strain.

    This diagnostic allows arbitrary Lame parameters, checks normal/shear
    coupling and symmetry, and does not select a geometry's material model.
    Physical admissibility (e.g. positive stiffness) is a separate question.
    Tolerances are applied after normalising by the largest matrix entry.
    """
    C = np.asarray(C, dtype=float)
    if C.shape != (3, 3) or not np.isfinite(C).all():
        return False
    scale = np.max(np.abs(C))
    if scale == 0:
        return True
    lam, mu = C[0, 1], (C[0, 0] - C[0, 1])/2
    isotropic = np.array([[lam+2*mu, lam, 0.],
                          [lam, lam+2*mu, 0.], [0., 0., mu]])
    return bool(np.allclose(C/scale, isotropic/scale, rtol=rtol, atol=atol))

def calc_IsoEffProperties(C):
    iso = check_isotropy(C)
    v = C[0][1]/C[0][0]  #1/(K[0][0]/K[0][1]+1)
    E = (C[0][0]**2 - C[0][1]**2)/C[0][0]  #(K[0][0]*((1+v)*(1-2*v)))/(1-v)
    return E, v, iso

def calc_ZenerRatio(C):
    Z = 2*C[2][2]/(C[0][0]-C[0][1])
    return Z

def calc_anisoParams(C=None, S=None):
    """Return lambda, chi for supplied in-plane compliance in order (1,2,6).

    This algebra does not infer constraint, reduce plane stress, or map
    specimen/source axes. Supply the already appropriate C or S explicitly.
    The second return is Ortega's rho (called chi to avoid density ambiguity).
    """
    if (C is None) == (S is None):
        raise ValueError("Supply exactly one of C or S.")
    if C is not None:
        S = calc_Compliance(C)
    S = np.asarray(S, dtype=float)
    if S.shape != (3, 3) or not np.isfinite(S).all() or S[0, 0] <= 0 or S[1, 1] <= 0:
        raise ValueError("S must be a finite 3-by-3 compliance with positive axial entries.")
    lambda_aniso = S[0,0]/S[1,1]
    chi_aniso = (2*S[0,1] + S[2,2])/(2*np.sqrt(S[0,0]*S[1,1]))
    return lambda_aniso, chi_aniso


def plot_IsotropyVariation(Ks, stiff=False, inplane=False, properties=False, zener=False, paper=False):
    K11, K33, K13, K23 = [], [], [], []
    for K in Ks[1:]:
        K11.append((K[0][0] - Ks[0][0][0])/Ks[0][0][0])
        K33.append((K[2][2] - Ks[0][2][2])/Ks[0][2][2])
        K13.append(K[0][2]/Ks[0][0][0])
        K23.append(K[1][2]/Ks[0][0][0])
    
    K_22_11p = Ks[0][1][1] / Ks[0][0][0]
    K_11_12p = Ks[0][0][0] / Ks[0][0][1]
    K_12_33p = Ks[0][0][1] / Ks[0][2][2]
    K_13_11p = Ks[0][0][2] / Ks[0][0][0]
    K_23_11p = Ks[0][1][2] / Ks[0][0][0]
    Zp = calc_ZenerRatio(Ks[0])
    K_22_11s, K_11_12s, K_12_33s, K_13_11s, K_23_11s, Zs = [], [], [], [], [], []
    for K in Ks[1:]:
        K_22_11s.append(K[1][1] / K[0][0])
        K_11_12s.append(K[0][0] / K[0][1])
        K_12_33s.append(K[0][1] / K[2][2])
        K_13_11s.append(K[0][2] / K[0][0])
        K_23_11s.append(K[1][2] / K[0][0])
        Zs.append(calc_ZenerRatio(K))
    
    Ep, vp, _ = calc_IsoEffProperties(Ks[0])
    vs, Es = [], []
    for K in Ks[1:]:
        Ed, vd, _ = calc_IsoEffProperties(K)
        vs.append(vd)
        Es.append(Ed)
    

    if stiff:
        fig1, (ax1, ax2) = plt.subplots(1, 2)
        fig1.set_figheight(5)
        fig1.set_figwidth(18)

        ax1.set_ylabel('$(C^{dis}-C^{p})/C^{p}$ (fraction)', fontsize=26, fontname="Times New Roman")
        ax1.set_xlabel('Dis Model No.', fontsize=26, fontname="Times New Roman")
        # ax1.set_ylim([-0.1, 0.1])
        # ax1.set_xticks([i+1 for i in range(len(K11))])
        ax1.plot([i+1 for i in range(len(K11))], K11, 'bx', markersize=10, label='$C_{11}$')
        ax1.plot([i+1 for i in range(len(K33))], K33, 'r^', markersize=10, label='$C_{66}$')
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.tick_params(axis='both', which='minor', labelsize=20)
        # ax1.grid()
        ax1.legend(prop={'size':18})

        ax2.set_ylabel('$C^{dis}/C^{p}_{11}$', fontsize=26, fontname="Times New Roman")
        ax2.set_xlabel('Dis Model No.', fontsize=26, fontname="Times New Roman")
        # ax2.set_ylim([-0.05, 0.05])
        # ax2.set_xticks([i+1 for i in range(len(K11))])
        ax2.plot([i+1 for i in range(len(K13))], K13, 'bx', markersize=10, label='$C^{dis}_{16}$')
        ax2.plot([i+1 for i in range(len(K23))], K23, 'r^', markersize=10, label='$C^{dis}_{26}$')
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(axis='both', which='minor', labelsize=20)
        # ax2.grid()
        ax2.legend(prop={'size':18})
        fig1.tight_layout()
    
    if inplane:    
        fig2, (ax3, ax4, ax5) = plt.subplots(1, 3)
        fig2.set_figheight(5)
        fig2.set_figwidth(22)

        ax3.set_ylabel('$C_{22}/C_{11}$ (ratio)', fontsize=26, fontname="Times New Roman")
        ax3.set_xlabel('Dis Model No.', fontsize=26, fontname="Times New Roman")
        ax3.tick_params(axis='both', which='major', labelsize=20)
        ax3.tick_params(axis='both', which='minor', labelsize=20)
        # ax3.set_ylim([-0.1, 0.1])
        # ax3.set_xticks([i+1 for i in range(len(K11))])
        ax3.axhline(y=K_22_11p, color='k', linestyle='--', label='Per')
        ax3.plot([i+1 for i in range(len(K_22_11s))], K_22_11s, 'bx', label='Dis')
        # ax3.grid()
        ax3.legend(prop={'size':18})

        ax4.set_ylabel('$C_{11}/C_{12}$ (ratio)', fontsize=26, fontname="Times New Roman")
        ax4.set_xlabel('Dis Model No.', fontsize=26, fontname="Times New Roman")
        ax4.tick_params(axis='both', which='major', labelsize=20)
        ax4.tick_params(axis='both', which='minor', labelsize=20)
        # ax4.set_ylim([-0.1, 0.1])
        # ax4.set_xticks([i+1 for i in range(len(K11))])
        ax4.axhline(y=K_11_12p, color='k', linestyle='--', label='Per')
        ax4.plot([i+1 for i in range(len(K_11_12s))], K_11_12s, 'bx', label='Dis')
        # ax4.grid()
        ax4.legend(prop={'size':18})

        ax5.set_ylabel('$C_{12}/C_{66}$ (ratio)', fontsize=26, fontname="Times New Roman")
        ax5.set_xlabel('Dis Model No.', fontsize=26, fontname="Times New Roman")
        ax5.tick_params(axis='both', which='major', labelsize=20)
        ax5.tick_params(axis='both', which='minor', labelsize=20)
        # ax5.set_ylim([-0.1, 0.1])
        # ax5.set_xticks([i+1 for i in range(len(K11))])
        ax5.axhline(y=K_12_33p, color='k', linestyle='--', label='Per')
        ax5.plot([i+1 for i in range(len(K_12_33s))], K_12_33s, 'bx', label='Dis')
        # ax5.grid()
        ax5.legend(prop={'size':18})
        fig2.tight_layout()

    if properties:        
        fig3, ax6 = plt.subplots(1, 1)
        fig3.set_figheight(5)
        fig3.set_figwidth(9)
        ax6_ = ax6.twinx()

        ax6.set_ylabel("Poisson's Ratio, $\\nu$ (blue)", fontsize=26, fontname="Times New Roman")
        ax6.set_xlabel('Dis Generated Model No.', fontsize=26, fontname="Times New Roman")
        ax6.tick_params(axis='both', which='major', labelsize=20)
        ax6.tick_params(axis='both', which='minor', labelsize=20)
        ax6_.set_ylabel("Young's Modulus, $E$ (red) [GPa]", fontsize=26, fontname="Times New Roman")
        ax6_.tick_params(axis='both', which='major', labelsize=20)
        ax6_.tick_params(axis='both', which='minor', labelsize=20)

        # ax6.set_ylim([vp-0.05, vp+0.05])
        # ax6_.set_ylim([min(Es)/1e9 - 5, max(Es)/1e9 + 5])
        # ax6.set_xticks([i+1 for i in range(len(K11))])
        ax6.axhline(y=vp, xmax=0.75, color='b', linestyle='--', label="Per Poisson's Ratio, $\\nu^{p}$ = %.2f" % vp)
        ax6.plot([i+1 for i in range(len(vs))], vs, 'bx', label="Dis Poisson's Ratio, $\\nu^{d}$")
        ax6_.axhline(y=Ep/1e9, xmin=0.25, color='r', linestyle='--', label="Per Young's Modulus, $E^{p}$ = %.2f GPa" % (Ep/1e9))
        ax6_.plot([i+1 for i in range(len(Es))], np.array(Es)/1e9, 'r^', label="Dis Young's Modulus, $E^{d}$")

        # ax6.grid()
        ax6.legend(prop={'size':18})
        ax6_.legend(prop={'size':18})
        fig3.tight_layout()

    if zener:
        fig4, ax7 = plt.subplots(1, 1)
        fig4.set_figheight(5)
        fig4.set_figwidth(9)

        ax7.set_ylabel('Zener Ratio, $a$', fontsize=26, fontname="Times New Roman")
        ax7.set_xlabel('Dis Model No.', fontsize=26, fontname="Times New Roman")
        ax7.tick_params(axis='both', which='major', labelsize=20)
        ax7.tick_params(axis='both', which='minor', labelsize=20)

        # ax7.set_ylim([Zp-0.15, Zp+0.15])
        # ax7.set_xticks([i+1 for i in range(len(K11))])
        ax7.axhline(y=Zp, color='k', linestyle='--', label='Per, $a^{p}$ = %.2f' % Zp)
        ax7.plot([i+1 for i in range(len(Zs))], Zs, 'bx', label='Dis, $a^{dis}$')

        # ax7.grid()
        ax7.legend(prop={'size':18})
        fig4.tight_layout()

    if paper:
        fig5, (ax8, ax9, ax10) = plt.subplots(1, 3)
        fig5.set_figheight(5)
        fig5.set_figwidth(23)

        ax8.set_ylabel('$C_{22}/C_{11}$', fontsize=26, fontname="Times New Roman")
        ax8.set_xlabel('Dis Model No.', fontsize=26, fontname="Times New Roman")
        ax8.tick_params(axis='both', which='major', labelsize=20)
        ax8.tick_params(axis='both', which='minor', labelsize=20)
        # ax8.set_ylim([-0.1, 0.1])
        # ax8.set_xticks([i+1 for i in range(len(K11))])
        ax8.axhline(y=K_22_11p, color='k', linestyle='--', linewidth=2.5, label='Per')
        ax8.plot([i+1 for i in range(len(K_22_11s))], K_22_11s, 'bx', markersize=10, label='Dis')
        # ax8.grid()
        ax8.legend(prop={'size':18})

        ax9.set_ylabel('$C^{*}/C_{11}$', fontsize=26, fontname="Times New Roman")
        ax9.set_xlabel('Dis Model No.', fontsize=26, fontname="Times New Roman")
        ax9.tick_params(axis='both', which='major', labelsize=20)
        ax9.tick_params(axis='both', which='minor', labelsize=20)
        # ax9.set_ylim([-0.05, 0.05])
        # ax9.set_xticks([i+1 for i in range(len(K11))])
        ax9.axhline(y=K_13_11p, color='k', linestyle='--', linewidth=2.5, label='Per')
        ax9.plot([i+1 for i in range(len(K_13_11s))], K_13_11s, 'bx', markersize=10, label='$C^{*}=C_{16}$')
        ax9.plot([i+1 for i in range(len(K_23_11s))], K_23_11s, 'r^', markersize=10, label='$C^{*}=C_{26}$')
        # ax9.grid()
        ax9.legend(prop={'size':18})

        ax10.set_ylabel('Zener Ratio, $a$', fontsize=26, fontname="Times New Roman")
        ax10.set_xlabel('Dis Model No.', fontsize=26, fontname="Times New Roman")
        ax10.tick_params(axis='both', which='major', labelsize=20)
        ax10.tick_params(axis='both', which='minor', labelsize=20)
        # ax10.set_ylim([Zp-0.15, Zp+0.15])
        # ax10.set_xticks([i+1 for i in range(len(K11))])
        ax10.axhline(y=Zp, color='k', linestyle='--', linewidth=2.5, label='Per, $a^{p}$ = %.2f' % Zp)
        ax10.plot([i+1 for i in range(len(Zs))], Zs, 'bx', markersize=10, label='Dis, $a^{dis}$')
        # ax10.grid()
        ax10.legend(prop={'size':18})
        fig5.tight_layout()
    
    plt.show()

