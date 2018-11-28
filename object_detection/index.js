import my_model, { downloadModel } from './model.js';

const image = document.document.getElementById("myCanvas");
      
const model = downloadModel();
console.log(model);
const boxes = my_model(image, model);
console.log(boxes);

boxes.forEach(box => {
  const {
    top, left, bottom, right, classProb, className,
  } = box;

  drawRect(left, top, right-left, bottom-top, `${className} ${classProb}`)
});
