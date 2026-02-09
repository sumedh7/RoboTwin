from ._base_task import Base_Task
from .utils import *
from .utils.create_actor import UnStableError
import sapien
import math
import glob
import json
import re
import time
from pathlib import Path
from copy import deepcopy

_MAX_STABLE_RETRIES = 5


class place_anyobject_stand(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        for attempt in range(_MAX_STABLE_RETRIES):
            try:
                super()._init_task_env_(**kwags)
                return
            except UnStableError as e:
                if attempt < _MAX_STABLE_RETRIES - 1:
                    print(f"  [place_anyobject_stand] Unstable object on attempt "
                          f"{attempt + 1}/{_MAX_STABLE_RETRIES}: {e}  — retrying")
                    self.close_env()
                    time.sleep(0.1)
                    # Bump the seed so the retry picks a different random pose / object
                    kwags["seed"] = kwags.get("seed", 0) + 10000 * (attempt + 1)
                else:
                    raise

    def load_actors(self):
        rand_pos = rand_pose(
            xlim=[-0.28, 0.28],
            ylim=[-0.05, 0.05],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 3, 0],
        )
        while abs(rand_pos.p[0]) < 0.2:
            rand_pos = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.05, 0.05],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 3, 0],
            )

        def _model_file_exists(modeldir, model_id):
            """Check that at least one model geometry file (glb or obj) exists."""
            modeldir = Path(modeldir)
            candidates = [
                modeldir / f"base{model_id}.glb",
                modeldir / f"textured{model_id}.obj",
                modeldir / "collision" / f"base{model_id}.glb",
                modeldir / "collision" / f"textured{model_id}.obj",
            ]
            return any(c.exists() for c in candidates)

        def _model_data_valid(json_path):
            """Check that the JSON is loadable and contains the required 'scale' key."""
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                return "scale" in data
            except Exception:
                return False

        def get_available_model_ids(modelname):
            asset_path = os.path.join("assets/objects", modelname)
            json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))

            available_ids = []
            for file in json_files:
                base = os.path.basename(file)
                try:
                    idx = int(base.replace("model_data", "").replace(".json", ""))
                except ValueError:
                    continue
                # Verify that the JSON is valid and the model geometry file exists
                if _model_data_valid(file) and _model_file_exists(asset_path, idx):
                    available_ids.append(idx)

            return available_ids

        def get_all_object_names():
            """Discover all numbered object directories in assets/objects,
            excluding the display stand (074_displaystand) since it is used
            as the target placement surface."""
            objects_dir = "assets/objects"
            pattern = re.compile(r"^\d{3}_")
            excluded = {"074_displaystand"}

            object_names = []
            for entry in sorted(os.listdir(objects_dir)):
                full_path = os.path.join(objects_dir, entry)
                if os.path.isdir(full_path) and pattern.match(entry) and entry not in excluded:
                    # Only include objects that have at least one fully valid model
                    if get_available_model_ids(entry):
                        object_names.append(entry)

            return object_names

        object_list = get_all_object_names()
        if not object_list:
            raise ValueError("No valid objects found in assets/objects")

        self.selected_modelname = np.random.choice(object_list)
        available_model_ids = get_available_model_ids(self.selected_modelname)
        self.selected_model_id = np.random.choice(available_model_ids)
        self.object = create_actor(
            scene=self,
            pose=rand_pos,
            modelname=self.selected_modelname,
            convex=True,
            model_id=self.selected_model_id,
        )
        if self.object is None or self.object.config is None:
            raise ValueError(
                f"Failed to load object {self.selected_modelname} "
                f"(model_id={self.selected_model_id}): actor or model data is None"
            )
        self.object.set_mass(0.05)

        object_pos = self.object.get_pose()
        if object_pos.p[0] > 0:
            xlim = [0.0, 0.05]
        else:
            xlim = [-0.05, 0.0]
        target_rand_pos = rand_pose(
            xlim=xlim,
            ylim=[-0.15, -0.1],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 6, 0],
        )
        while ((object_pos.p[0] - target_rand_pos.p[0])**2 + (object_pos.p[1] - target_rand_pos.p[1])**2) < 0.01:
            target_rand_pos = rand_pose(
                xlim=xlim,
                ylim=[-0.15, -0.1],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 6, 0],
            )
        id_list = [0, 1, 2, 3, 4]
        self.displaystand_id = np.random.choice(id_list)
        self.displaystand = create_actor(
            scene=self,
            pose=target_rand_pos,
            modelname="074_displaystand",
            convex=True,
            model_id=self.displaystand_id,
        )

        self.object.set_mass(0.01)
        self.displaystand.set_mass(0.01)

        self.add_prohibit_area(self.displaystand, padding=0.05)
        self.add_prohibit_area(self.object, padding=0.1)

    def play_once(self):
        # Determine which arm to use based on object's x position
        arm_tag = ArmTag("right" if self.object.get_pose().p[0] > 0 else "left")

        # Grasp the object with specified arm
        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1))
        # Lift the object up by 0.06 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.06))

        # Get the target pose from display stand's functional point
        displaystand_pose = self.displaystand.get_functional_point(0)

        # Place the object onto the display stand with free constraint
        self.move(
            self.place_actor(
                self.object,
                arm_tag=arm_tag,
                target_pose=displaystand_pose,
                constrain="free",
                pre_dis=0.07,
            ))

        # Store information about the objects and arm used in the info dictionary
        self.info["info"] = {
            "{A}": f"{self.selected_modelname}/base{self.selected_model_id}",
            "{B}": f"074_displaystand/base{self.displaystand_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        object_pose = self.object.get_pose().p
        displaystand_pose = self.displaystand.get_pose().p
        eps1 = 0.03
        return (np.all(abs(object_pose[:2] - displaystand_pose[:2]) < np.array([eps1, eps1]))
                and self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open())
