import React from "react";

const Grid = () => {
  return (
    <ul class="grid">
      <li class="item item--big">
        <div class="item__inner">
          <img class="item__content" src="https://picsum.photos/536/354" />
        </div>
      </li>
      <li class="item">
        <div class="item__inner">
          <img class="item__content" src="https://picsum.photos/536/354" />
        </div>
      </li>
      <li class="item">
        <div class="item__inner">
          <img class="item__content" src="https://picsum.photos/536/354" />
        </div>
      </li>
      <li class="item item--big item--right">
        <div class="item__inner">
          <img class="item__content" src="https://picsum.photos/536/354" />
        </div>
      </li>
      <li class="item">
        <div class="item__inner">
          <img class="item__content" src="https://picsum.photos/536/354" />
        </div>
      </li>
      <li class="item">
        <div class="item__inner">
          <img class="item__content" src="https://picsum.photos/536/354" />
        </div>
      </li>
    </ul>
  );
};

export default Grid;
