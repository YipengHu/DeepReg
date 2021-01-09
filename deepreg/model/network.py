from typing import Dict

import tensorflow as tf

from deepreg.model import layer, layer_util
from deepreg.registry import REGISTRY


def dict_without(d: dict, key) -> dict:
    copied = d.copy()
    copied.pop(key)
    return copied


class RegistrationModel(tf.keras.Model):
    def __init__(
        self,
        moving_image_size: tuple,
        fixed_image_size: tuple,
        index_size: int,
        labeled: bool,
        batch_size: int,
        config: dict,
    ):
        super().__init__()
        self.moving_image_size = moving_image_size
        self.fixed_image_size = fixed_image_size
        self.index_size = index_size
        self.labeled = labeled
        self.batch_size = batch_size
        self.config = config

        self._model = self.build_model()
        self.build_loss()

    def build_model(self):
        raise NotImplementedError

    def build_inputs(self) -> Dict[str, tf.keras.layers.Input]:
        """
        Build input tensors.

        :return: tuple
        """
        # (batch, m_dim1, m_dim2, m_dim3, 1)
        moving_image = tf.keras.Input(
            shape=self.moving_image_size,
            batch_size=self.batch_size,
            name="moving_image",
        )
        # (batch, f_dim1, f_dim2, f_dim3, 1)
        fixed_image = tf.keras.Input(
            shape=self.fixed_image_size,
            batch_size=self.batch_size,
            name="fixed_image",
        )
        # (batch, index_size)
        indices = tf.keras.Input(
            shape=(self.index_size,),
            batch_size=self.batch_size,
            name="indices",
        )

        if not self.labeled:
            return dict(
                moving_image=moving_image, fixed_image=fixed_image, indices=indices
            )

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        moving_label = tf.keras.Input(
            shape=self.moving_image_size,
            batch_size=self.batch_size,
            name="moving_label",
        )
        # (batch, m_dim1, m_dim2, m_dim3, 1)
        fixed_label = tf.keras.Input(
            shape=self.fixed_image_size,
            batch_size=self.batch_size,
            name="fixed_label",
        )
        return dict(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_label=moving_label,
            fixed_label=fixed_label,
            indices=indices,
        )

    def concat_images(self, moving_image, fixed_image, moving_label=None):
        images = []

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        moving_image = tf.expand_dims(moving_image, axis=4)
        moving_image = layer_util.resize3d(
            image=moving_image, size=self.fixed_image_size
        )
        images.append(moving_image)

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        fixed_image = tf.expand_dims(fixed_image, axis=4)
        images.append(fixed_image)

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        if moving_label is not None:
            moving_label = tf.expand_dims(moving_label, axis=4)
            moving_label = layer_util.resize3d(
                image=moving_label, size=self.fixed_image_size
            )
            images.append(moving_label)

        # (batch, f_dim1, f_dim2, f_dim3, 2 or 3)
        images = tf.concat(images, axis=4)
        return images

    def _build_loss(self, name: str, inputs_dict):
        # build loss
        config = self.config["loss"][name]
        loss_cls = REGISTRY.build_loss(config=dict_without(d=config, key="weight"))
        loss = loss_cls(**inputs_dict)
        weighted_loss = loss * config["weight"]

        # add loss
        self._model.add_loss(weighted_loss)

        # add metric
        self._model.add_metric(
            loss, name=f"loss/{name}_{loss_cls.name}", aggregation="mean"
        )
        self._model.add_metric(
            weighted_loss,
            name=f"loss/{name}_{loss_cls.name}_weighted",
            aggregation="mean",
        )

    def build_loss(self):
        raise NotImplementedError

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs, training=training, mask=mask)

    def postprocess(self, inputs, outputs):
        raise NotImplementedError


@REGISTRY.register_model(name="ddf")
class DDFModel(RegistrationModel):
    def build_model(self):
        # build inputs
        inputs = self.build_inputs()
        moving_image = inputs["moving_image"]
        fixed_image = inputs["fixed_image"]

        # build ddf
        backbone_inputs = self.concat_images(moving_image, fixed_image)
        backbone = REGISTRY.build_backbone(
            config=self.config["backbone"],
            default_args=dict(
                image_size=self.fixed_image_size,
                out_channels=3,
                out_kernel_initializer="zeros",
                out_activation=None,
            ),
        )
        # save backbone in case of affine to retrieve theta
        self._backbone = backbone

        # (f_dim1, f_dim2, f_dim3, 3)
        ddf = backbone(inputs=backbone_inputs)

        # build outputs
        warping = layer.Warping(fixed_image_size=self.fixed_image_size)
        # (f_dim1, f_dim2, f_dim3, 3)
        pred_fixed_image = warping(inputs=[ddf, moving_image])

        if not self.labeled:
            outputs = dict(ddf=ddf, pred_fixed_image=pred_fixed_image)
            self._inputs = inputs
            self._outputs = outputs
            return tf.keras.Model(inputs=inputs, outputs=outputs)

        # (f_dim1, f_dim2, f_dim3, 3)
        moving_label = inputs["moving_label"]
        pred_fixed_label = warping(inputs=[ddf, moving_label])

        outputs = dict(
            ddf=ddf,
            pred_fixed_image=pred_fixed_image,
            pred_fixed_label=pred_fixed_label,
        )
        self._inputs = inputs
        self._outputs = outputs
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def build_loss(self):
        fixed_image = self._inputs["fixed_image"]
        ddf = self._outputs["ddf"]
        pred_fixed_image = self._outputs["pred_fixed_image"]

        # ddf
        self._build_loss(name="regularization", inputs_dict=dict(inputs=ddf))

        # image
        self._build_loss(
            name="image", inputs_dict=dict(y_true=fixed_image, y_pred=pred_fixed_image)
        )

        # label
        if self.labeled:
            fixed_label = self._inputs["fixed_label"]
            pred_fixed_label = self._outputs["pred_fixed_label"]
            self._build_loss(
                name="label",
                inputs_dict=dict(y_true=fixed_label, y_pred=pred_fixed_label),
            )

    def postprocess(self, inputs, outputs):
        # each value is (tensor, normalize, on_label), where
        # - normalize = True if the tensor need to be normalized to [0, 1]
        # - on_label = True if the tensor depends on label
        indices = inputs["indices"]
        processed = dict(
            moving_image=(inputs["moving_image"], True, False),
            fixed_image=(inputs["fixed_image"], True, False),
            ddf=(outputs["ddf"], True, False),
            pred_fixed_image=(outputs["pred_fixed_image"], True, False),
        )

        # save theta for affine model
        if hasattr(self._backbone, "theta"):
            processed["theta"] = (self._backbone.theta, None, None)

        if not self.labeled:
            return indices, processed

        processed = {
            **dict(
                moving_label=(inputs["moving_label"], False, True),
                fixed_label=(inputs["fixed_label"], False, True),
                pred_fixed_label=(outputs["pred_fixed_label"], False, True),
            ),
            **processed,
        }

        return indices, processed


@REGISTRY.register_model(name="dvf")
class DVFModel(RegistrationModel):
    def build_model(self):
        # build inputs
        inputs = self.build_inputs()
        moving_image = inputs["moving_image"]
        fixed_image = inputs["fixed_image"]

        # build ddf
        backbone_inputs = self.concat_images(moving_image, fixed_image)
        backbone = REGISTRY.build_backbone(
            config=self.config["backbone"],
            default_args=dict(
                image_size=self.fixed_image_size,
                out_channels=3,
                out_kernel_initializer="zeros",
                out_activation=None,
            ),
        )
        dvf = backbone(inputs=backbone_inputs)
        ddf = layer.IntDVF(fixed_image_size=self.fixed_image_size)(dvf)

        # build outputs
        warping = layer.Warping(fixed_image_size=self.fixed_image_size)
        # (f_dim1, f_dim2, f_dim3, 3)
        pred_fixed_image = warping(inputs=[ddf, moving_image])

        if not self.labeled:
            outputs = dict(dvf=dvf, ddf=ddf, pred_fixed_image=pred_fixed_image)
            self._inputs = inputs
            self._outputs = outputs
            return tf.keras.Model(inputs=inputs, outputs=outputs)

        # (f_dim1, f_dim2, f_dim3, 3)
        moving_label = inputs["moving_label"]
        pred_fixed_label = warping(inputs=[ddf, moving_label])

        outputs = dict(
            dvf=dvf,
            ddf=ddf,
            pred_fixed_image=pred_fixed_image,
            pred_fixed_label=pred_fixed_label,
        )
        self._inputs = inputs
        self._outputs = outputs
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def build_loss(self):
        fixed_image = self._inputs["fixed_image"]
        ddf = self._outputs["ddf"]
        pred_fixed_image = self._outputs["pred_fixed_image"]

        # ddf
        self._build_loss(name="regularization", inputs_dict=dict(inputs=ddf))

        # image
        self._build_loss(
            name="image", inputs_dict=dict(y_true=fixed_image, y_pred=pred_fixed_image)
        )

        # label
        if self.labeled:
            fixed_label = self._inputs["fixed_label"]
            pred_fixed_label = self._outputs["pred_fixed_label"]
            self._build_loss(
                name="label",
                inputs_dict=dict(y_true=fixed_label, y_pred=pred_fixed_label),
            )

    def postprocess(self, inputs, outputs):
        # each value is (tensor, normalize, on_label), where
        # - normalize = True if the tensor need to be normalized to [0, 1]
        # - on_label = True if the tensor depends on label
        indices = inputs["indices"]
        processed = dict(
            moving_image=(inputs["moving_image"], True, False),
            fixed_image=(inputs["fixed_image"], True, False),
            dvf=(outputs["dvf"], True, False),
            ddf=(outputs["ddf"], True, False),
            pred_fixed_image=(outputs["pred_fixed_image"], True, False),
        )

        if not self.labeled:
            return indices, processed

        processed = {
            **dict(
                moving_label=(inputs["moving_label"], False, True),
                fixed_label=(inputs["fixed_label"], False, True),
                pred_fixed_label=(outputs["pred_fixed_label"], False, True),
            ),
            **processed,
        }

        return indices, processed


@REGISTRY.register_model(name="conditional")
class ConditionalModel(RegistrationModel):
    def build_model(self):
        assert self.labeled

        # build inputs
        inputs = self.build_inputs()
        moving_image = inputs["moving_image"]
        fixed_image = inputs["fixed_image"]
        moving_label = inputs["moving_label"]

        # build ddf
        backbone_inputs = self.concat_images(moving_image, fixed_image, moving_label)
        backbone = REGISTRY.build_backbone(
            config=self.config["backbone"],
            default_args=dict(
                image_size=self.fixed_image_size,
                out_channels=1,
                out_kernel_initializer="glorot_uniform",
                out_activation="sigmoid",
            ),
        )
        # (batch, f_dim1, f_dim2, f_dim3)
        pred_fixed_label = backbone(inputs=backbone_inputs)
        pred_fixed_label = tf.squeeze(pred_fixed_label, axis=4)

        outputs = dict(pred_fixed_label=pred_fixed_label)
        self._inputs = inputs
        self._outputs = outputs
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def build_loss(self):
        fixed_label = self._inputs["fixed_label"]
        pred_fixed_label = self._outputs["pred_fixed_label"]

        self._build_loss(
            name="label",
            inputs_dict=dict(y_true=fixed_label, y_pred=pred_fixed_label),
        )

    def postprocess(self, inputs, outputs):

        # each value is (tensor, normalize, on_label), where
        # - normalize = True if the tensor need to be normalized to [0, 1]
        # - on_label = True if the tensor depends on label
        indices = inputs["indices"]
        processed = dict(
            moving_image=(inputs["moving_image"], True, False),
            fixed_image=(inputs["fixed_image"], True, False),
            pred_fixed_label=(outputs["pred_fixed_label"], True, True),
            moving_label=(inputs["moving_label"], False, True),
            fixed_label=(inputs["fixed_label"], False, True),
        )

        return indices, processed
