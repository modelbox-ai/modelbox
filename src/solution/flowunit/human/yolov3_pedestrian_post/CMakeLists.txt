cmake_minimum_required(VERSION 3.10)

set(UNIT_DEVICE "cpu")
set(UNIT_NAME "pedestrian_yolobox")

set(YOLO_TOML ${CMAKE_CURRENT_BINARY_DIR}/${UNIT_NAME}.toml)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${UNIT_NAME}.toml.in ${YOLO_TOML} @ONLY)
install(FILES 
    ${YOLO_TOML} DESTINATION ${SOLUTION_FLOWUNIT_TRAFFIC_DIR}
    COMPONENT solution
)

# driver test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${UNIT_NAME}.toml.in  ${TEST_SOLUTION_DRIVERS_DIR}/${UNIT_NAME}.toml @ONLY)