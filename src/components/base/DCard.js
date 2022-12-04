// import { useState } from "react";

// function DraggableCard(props) {
//   const [position, setPosition] = useState({ x: 0, y: 0 });

//   function handleDragStart(e) {
//     // Set the initial position of the card when the user starts dragging it.
//     setPosition({
//       x: e.clientX,
//       y: e.clientY,
//     });
//   }

//   function handleDrag(e) {
//     // Update the position of the card as the user drags it.
//     setPosition({
//       x: e.clientX,
//       y: e.clientY,
//     });
//   }

//   function handleDragEnd() {
//     // Reset the position of the card when the user stops dragging it.
//     setPosition({
//       x: 0,
//       y: 0,
//     });
//   }

//   return (
//     <div
//       className="draggable-card"
//       style={{
//         // Use the position state to set the style of the card so that it
//         // appears to move as the user drags it.
//         transform: `translate(${position.x}px, ${position.y}px)`,
//       }}
//       onDragStart={handleDragStart}
//       onDrag={handleDrag}
//       onDragEnd={handleDragEnd}
//     >
//       {props.children}
//     </div>
//   );
// }

import { useDrag } from "dnd";

function DraggableCard(props) {
  const [, drag] = useDrag({
    item: { type: "CARD" },
  });

  return (
    <div
      className="draggable-card"
      ref={drag}
      style={{
        width: 100,
        height: 400,
      }}
    >
      {props.children}
    </div>
  );
}

export default DraggableCard;
