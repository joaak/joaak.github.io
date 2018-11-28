import my_model, { downloadModel } from './model.js';

const model = downloadModel();
const boxes = my_model(model);

boxes.forEach(box => {
  const {
    top, left, bottom, right, classProb, className,
  } = box;

  drawRect(left, top, right-left, bottom-top, `${className} ${classProb}`)
});
