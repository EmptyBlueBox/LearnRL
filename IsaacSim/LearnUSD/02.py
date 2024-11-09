from pxr import Usd, Vt

stage = Usd.Stage.Open('franka_alt_fingers.usd')

xform = stage.GetPrimAtPath('/')

xform.GetPropertyNames()