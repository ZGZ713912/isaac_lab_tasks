"""Reference static field USDs without importing their PhysicsScene."""

import math

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
from isaaclab.terrains import TerrainImporter
import isaaclab.sim as sim_utils


class FieldTerrainImporter(TerrainImporter):
    def import_usd(self, name, usd_path):
        self.field_path = self.cfg.prim_path + "/" + name
        stage = sim_utils.get_current_stage()
        if UsdGeom.GetStageUpAxis(stage) != "Z" or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
            raise ValueError("Field destination stage must use metres and Z-up")
        with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
            field = UsdGeom.Xform.Define(stage, self.field_path).GetPrim()
            if not field.GetReferences().AddReference(usd_path):
                raise RuntimeError(f"Could not reference field: {usd_path}")
            for prim in list(Usd.PrimRange(field)):
                if prim.IsA(UsdPhysics.Scene):
                    prim.SetActive(False)
            # Retain the task's original friction and combine mode on the field.
            self.material_path = self.cfg.prim_path + "/PhysicsMaterial"
            material = self.cfg.physics_material.func(self.material_path, self.cfg.physics_material)
            UsdShade.MaterialBindingAPI.Apply(field).Bind(UsdShade.Material(material),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")
        self.terrain_prim_paths.append(self.field_path)
