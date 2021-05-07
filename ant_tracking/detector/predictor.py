from detectron2.engine import DefaultPredictor
import cv2 as cv
import torch

class Predictor(DefaultPredictor):
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more fancy, please refer to its source code as examples
    to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        super(Predictor,self).__init__(cfg)

    def batch_prediction(self,batch):
        """
        Args:
            batch list : str list with paths of batch frames

        Returns:
            predictions [dics]:
                the output of the model for the batch. each dic is for one frame
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            inputs = []
            for im in batch:
                #read image
                im = cv.imread(im, cv.IMREAD_COLOR)
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    im = im[:, :, ::-1]    
                
                height, width = im.shape[:2]
                #transform image to tensor
                image = self.aug.get_transform(im).apply_image(im)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                # prepare input for model
                inputs.append({"image": image, "height": height, "width": width})
            
            # feed input to model for batch prediction    
            predictions = self.model(inputs)                
            return predictions
