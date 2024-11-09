from pxr import Usd, Vt, UsdGeom

stage = Usd.Stage.Open('asset/HelloWorld.usda')

root = stage.GetPrimAtPath('/hello')
sphere = stage.GetPrimAtPath('/hello/world')

print(root.GetPropertyNames())
print(root.GetProperties())
print(root.GetAttributes())
print(root.GetRelationships())

# Double radius
radiusAttr = sphere.GetAttribute('radius')
radiusAttr.Set(radiusAttr.Get() * 2)

# Double extent
extentAttr = sphere.GetAttribute('extent')
extentAttr.Set(extentAttr.Get() * 2)

# Set display color
sphereSchema = UsdGeom.Sphere(sphere)
color = sphereSchema.GetDisplayColorAttr()
color.Set([(0,0,1)])

# Print the Ball
print(stage.GetRootLayer().ExportToString())


# Print the hierarchy of h1.usd
stage = Usd.Stage.Open('asset/h1.usd')
print([x.GetTypeName() for x in Usd.Stage.Traverse(stage)])

# Print the hierarchy of g1.usd
stage = Usd.Stage.Open('asset/g1.usd')
print([x.GetTypeName() for x in Usd.Stage.Traverse(stage)])
