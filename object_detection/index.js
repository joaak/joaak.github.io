import my_model, { downloadModel } from './model.js';

const model = await downloadModel();
const boxes = await my_model(model);

boxes.forEach(box => {
  const {
    top, left, bottom, right, classProb, className,
  } = box;

  drawRect(left, top, right-left, bottom-top, `${className} ${classProb}`)
});
