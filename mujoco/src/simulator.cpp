// standard stuff
#include "stdio.h"
#include "string.h"
#include <iostream>
#include <sys/types.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

// eigen
#include "Eigen/Dense"
#include "Eigen/Core"

// mujoco stuff
#include "mujoco.h"
#include "GLFW/glfw3.h"

// socket stuff
#define PORT 8080
#define MAXLINE 1000
#define mjUSEDOUBLE

// ********************************** Mujoco Structs *************************************** //

//simulation end time
char path[] = "../../models/";
char xmlfile[] = "achilles.xml";

// MuJoCo data structures
mjModel *m = NULL;                  // MuJoCo model
mjData *d = NULL;                   // MuJoCo data
mjContact *c = NULL;                // MuJoCo contact
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context
mjfSensor sens;

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// holders of one step history of time and position to calculate dertivatives
mjtNum position_history = 0;
mjtNum previous_time = 0;

// controller related variables
float_t ctrl_update_freq = 100;
mjtNum last_update = 0.0;
mjtNum ctrl;

// ********************************** GLFW Stuff *************************************** //

// keyboard callback
void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods) {
    // backspace: reset simulation
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}

// mouse button callback
void mouse_button(GLFWwindow *window, int button, int act, int mods) {
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}

// mouse move callback
void mouse_move(GLFWwindow *window, double xpos, double ypos) {
    // no buttons down: nothing to do
    if (!button_left && !button_middle && !button_right)
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right)
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (button_left)
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow *window, double xoffset, double yoffset) {
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

// ********************************** Mujoco Stuff *************************************** //

void set_torque_control(const mjModel *m, int actuator_no, int flag) {
    if (flag == 0)
        m->actuator_gainprm[10 * actuator_no + 0] = 0;
    else
        m->actuator_gainprm[10 * actuator_no + 0] = 1;
}

void set_position_servo(const mjModel *m, int actuator_no, double kp) {
    m->actuator_gainprm[10 * actuator_no + 0] = kp;
    m->actuator_biasprm[10 * actuator_no + 1] = -kp;
}

void set_velocity_servo(const mjModel *m, int actuator_no, double kv) {
    m->actuator_gainprm[10 * actuator_no + 0] = kv;
    m->actuator_biasprm[10 * actuator_no + 2] = -kv;
}

void init_controller(const mjModel *m, mjData *d) {

}

void mycontroller(const mjModel *m, mjData *d) {

}

// *************************************** MAIN **************************************** //

int main(int argc, const char **argv) {

    // path to XML
    char xmlpath[100] = {};
    char datapath[100] = {};
    strcat(xmlpath, path);
    strcat(xmlpath, xmlfile);
    strcat(datapath, path);

    // load  model
    char error[1000] = "Could not load binary model";
    m = mj_loadXML(xmlpath, 0, error, 1000);

    //create data
    d = mj_makeData(m);

    /* ------------------------- GLFW Init -------------------------*/
    // initialize GLFW
    if (!glfwInit()) {
        mju_error("Could not initialize GLFW");
    }

    // create window, make OpenGL context current, request v-sync
    GLFWwindow *window = glfwCreateWindow(1244, 700, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);                // space for 2000 objects
    mjr_makeContext(m, &con, mjFONTSCALE_150);   // model-specific context

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    double arr_view[] = {89.608063, -11.588379, 2, 0.000000, 0.000000,
                         0.500000}; //view the left side (for ll, lh, left_side)
    cam.azimuth = arr_view[0];
    cam.elevation = arr_view[1];
    cam.distance = arr_view[2];
    cam.lookat[0] = arr_view[3];
    cam.lookat[1] = arr_view[4];
    cam.lookat[2] = arr_view[5];

    // install control callback
    mjcb_control = mycontroller;
    init_controller(m, d);    

    // initialize random seed
    srand(time(NULL));

    /* ------------------------- Socket -------------------------*/ 
    // Setup the socket to communicate between the simulator and the controller
    int *new_socket = new int;
    int valread;
    struct sockaddr_in serv_addr;

    // // [receive - RX] Torques and horizon states: TODO: Fill in
    // scalar_t RX_torques[23] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0};

    // // [to send - TX] States: time[1], pos[3], quat[4], vel[3], omega[3], contact[1], leg (pos,vel)[2], flywheel speed [3]
    // scalar_t TX_state[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // if ((*new_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    //     printf("\n Socket creation error \n");
    //     return -1;
    // }
    // serv_addr.sin_family = AF_INET;
    // serv_addr.sin_port = htons(PORT);
    
    // // Convert IPv4 and IPv6 addresses from text to binary form
    // if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
    //     printf("\nInvalid address/ Address not supported \n");
    //     return -1;
    // }
    // if (connect(*new_socket, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
    //     printf("\nConnection Failed \n");
    //     return -1;
    // }

    /* ------------------------- Simulation -------------------------*/ 
    
    // Set the initial condition [pos, orientation, vel, angular rate]
    d->qpos[0] = 0.0;
    d->qpos[1] = 0.0;
    d->qpos[2] = 6.0;
    d->qpos[4] = 0.0;
    d->qpos[5] =  0.0;
    d->qpos[6] = 0.0;
    d->qvel[0] = 0.0;
    d->qvel[1] = 0.0;
    d->qvel[2] = 0.0;
    d->qvel[3] = 0.0;
    d->qvel[4] = 0.0;
    d->qvel[5] = 0.0;

    // get framebuffer viewport
    mj_step(m, d); // populate state info
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
 
    // update scene and render
    //cam.lookat[0] = d->qpos[0];
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
 
    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);
 
    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
    sleep(1);
    c = d->contact;
 
    // Instantiate perturbation object
    mjvPerturb* pert = new mjvPerturb();
    pert->select = 1;
    pert->active = 1;
    opt.flags[mjVIS_PERTFORCE] = 1;
    int iter = 0;


    // terminate GLFW (crashes with Linux NVidia drivers)
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif

    std::cout << "Hello, World!\n";
    return 0;
}