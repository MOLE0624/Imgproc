import React, { useContext, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import {
  DragDropContext,
  Draggable,
  Droppable,
  useSensor,
} from "react-beautiful-dnd";
import Button from "react-bootstrap/Button";
import { disableBodyScroll } from "body-scroll-lock";
import { TestList } from "../../TestList";
import Card from "./Card";
import "../../index.scss";

window.addEventListener("error", (e) => {
  if (
    e.message ===
      "ResizeObserver loop completed with undelivered notifications." ||
    e.message === "ResizeObserver loop limit exceeded"
  ) {
    e.stopImmediatePropagation();
  }
});

function Board() {
  const [testList, setTestList] = useState(TestList);
  const [selected, setSelected] = useState({ item: "", active: false });
  const [isDelete, setDelete] = useState(0);

  // const mouseSensor = useSensor();

  const bodyScrollLock = require("body-scroll-lock");
  const disableBodyScroll = bodyScrollLock.disableBodyScroll;
  disableBodyScroll("body");

  // const HeightPreservingItem = React.useMemo(() => {
  //   return ({ children, ...props }) => {
  //     return (
  //       // the height is necessary to prevent the item container from collapsing, which confuses Virtuoso measurements
  //       <div {...props} style={{ height: props["500px"] || undefined }}>
  //         {children}
  //       </div>
  //     );
  //   };
  // }, []);
  const [name, setName] = React.useState("");

  function handleChange(event) {
    setName(event.target.value);
  }

  const handleDelete = (id) => {
    console.log(typeof id);
    console.log(typeof testList[0].id);
    const newList = testList.filter((section) => section.id !== id);
    console.log(id);
    console.log("new: ", newList);
    setTestList(newList);
  };

  function handleAdd() {
    const newList = testList.slice();
    newList.push({ id: uuidv4(), name: "hello" });
    setTestList(newList);
  }

  const onDragEndTest = (result) => {
    const items = [...testList];
    const deleteItem = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, deleteItem[0]);

    setTestList(items);
  };

  const handleClick = (section) => {
    // setSelected(section);
    // console.log(`selected: ${selected.name}`);
    if (section === selected.item) {
      setSelected({ item: selected.item, active: !selected.active });
    } else {
      setSelected({ item: section, active: true });
    }
    console.log(
      `${section.id} ${section.name} was clicked. ${selected.active}`
    );
  };

  const AddItem = ({ name, onChange, onAdd }) => (
    <div>
      <input type="text" value={name} onChange={onChange} />
      <button type="button" onClick={onAdd}>
        Add
      </button>
    </div>
  );

  return (
    <div style={{ overflow: "auto" }}>
      <div className="trello-section-title">
        {/* <h1>ドラッグアンドドロップ</h1> */}
        <DragDropContext onDragStart={handleClick} onDragEnd={onDragEndTest}>
          <div className="trello">
            <Droppable droppableId="droppableId">
              {(provided) => {
                return (
                  <>
                    <div
                      className="trello-section"
                      {...provided.droppableProps}
                      ref={provided.innerRef}
                      style={{ overflow: "auto" }}
                    >
                      {testList.map((section, index) => {
                        return (
                          <Draggable
                            draggableId={section.id}
                            index={index}
                            key={section.id}
                          >
                            {(provided, snapshot) => (
                              <div
                                ref={provided.innerRef}
                                {...provided.draggableProps}
                                {...provided.dragHandleProps}
                                style={{
                                  ...provided.draggableProps.style,
                                  opacity: snapshot.isDragging ? "0.5" : "1",
                                  color: snapshot.isDragging
                                    ? "green"
                                    : "white",
                                }}
                                // className={
                                //   snapshot.isDragging ? "is-dragging" : "card"
                                // }
                                onClick={() => handleClick(section)}
                              >
                                <Card
                                  id={section.id}
                                  name={section.name}
                                  selected={selected}
                                  handleDelete={handleDelete}
                                />
                              </div>
                            )}
                          </Draggable>
                        );
                      })}
                      <button type="submit" onClick={handleAdd}>
                        Submit
                      </button>
                      {provided.placeholder}
                    </div>
                  </>
                );
              }}
            </Droppable>
          </div>
        </DragDropContext>
      </div>
    </div>
  );
}

export default Board;
