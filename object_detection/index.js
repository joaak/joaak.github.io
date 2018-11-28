import my_model from './model.js';

// const model = downloadModel();
// console.log(model);
const boxes = my_model();
console.log(boxes);

boxes.forEach(box => {
  const {
    top, left, bottom, right, classProb, className,
  } = box;

  drawRect(left, top, right-left, bottom-top, `${className} ${classProb}`)
});
