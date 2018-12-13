import java.io.{BufferedWriter, FileWriter}

import breeze.linalg.DenseVector

import scala.collection.mutable.HashMap
import breeze.numerics.{pow, sqrt}

import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

import scala.util.control.Breaks


object Niharika_Gajam_Task1{


  def euclidean_dist(vect1 : Vector,vect2 : Vector):(Double)={


    var v1 = vect1.toDense
    var v2 = vect2//.toDense  this is always centroid and in dense vector form always


   var i =0
    var sz = v1.size
    var sum = 0.0
    while(i < sz){
      val x = v1.apply(i) - v2.apply(i)
      sum = sum + pow(x,2)
      i = i+1
    }

    //println("sum is " + sum.toString)
    sqrt(sum)
  }


  var centroids = HashMap.empty[Int, Vector]
  var clusters_set  =  HashMap.empty[Int, Set[Vector]]

  def square_euclid_dist(vect1 : Vector,vect2 : Vector):(Double)={


    var v1 = vect1.toDense
    var v2 = vect2//.toDense  this is always centroid and in dense vector form always

    var i =0
    var sz = v1.size
    var sum = 0.0
    while(i < sz){
      val x = v1.apply(i) - v2.apply(i)
      sum = sum + pow(x,2)
      i = i+1
    }

    //println("sum is " + sum.toString)
    sum
  }

  var esse_per_cluster = HashMap.empty[Int, Double]

  def WSSE_calculation():(Double)={

    var sum =0.toDouble
    centroids.foreach(c => {

      var sum_per_cluster = 0.toDouble
      var centroid_index = c._1
      var centroid_vector = c._2

      clusters_set(centroid_index).foreach(vect =>{

        sum_per_cluster = sum_per_cluster + square_euclid_dist(vect,centroid_vector)
      })

      sum = sum + sum_per_cluster

      if(!esse_per_cluster.contains(centroid_index))
        esse_per_cluster(centroid_index) = sum_per_cluster
    })
    sum.toDouble
  }

  def sum_vectors(vect1 : Vector,vect2 : Vector):(Vector)={
    var v1 = vect1.toDense
    var v2 = vect2.toDense
    val bv1 = new DenseVector(v1.toArray)
    val bv2 = new DenseVector(v2.toArray)
    val vectout = Vectors.dense((bv1 + bv2).toArray)
    vectout.toSparse
  }

  def calculate_centroid_vect(s : Set[Vector]):(Vector)={

    var col = s.toList
    var sz = col.size
    var sum_vect = col(0)

    var i= 1
    while(i < sz){
      sum_vect = sum_vectors(sum_vect,col(i))
      i= i+1
    }

    val centroid_vect : Vector = Vectors.dense(sum_vect.toArray.map(x => x/sz))

    centroid_vect//.toSparse
  }
  var top_freq_words = HashMap.empty[Int, Set[String]]

  def get_word(input_file : RDD[Seq[String]],hashingTF: HashingTF,idx : Int):(String)={
    var ans = "".toString


    val str_ans =  input_file.map(ip=>{
      ip.toList.foreach(s=>{
        if(hashingTF.indexOf(s.toString) == idx){
          //println(s.toString)
          ans = s
        }
      })
      (ans,1)
    }).groupByKey().map(x=>(x._1,1)).collect().toArray

    var str =  str_ans(0)._1


    var ind=0
    while(ind < str_ans.size){
      if(!str_ans(ind)._1.isEmpty){
        str = str_ans(ind)._1
      }
      ind = ind + 1
    }

    str
  }

  def main(args: Array[String]): Unit = {

    val spark_config = new SparkConf().setAppName("Kmeans Clustering").setMaster("local[1]")
    val spark_context = new SparkContext(spark_config)



    val input_file: RDD[Seq[String]] = spark_context.textFile(args(0)).map(_.split(" ").toSeq)


    var feature = args(1).toString //"T".toString  //arg(1)
    val N = args(2).toInt // 5.toInt  //arg(2)
    var iterations = args(3).toInt //20.toInt //arg(3)


    // word count method first , should be easy to work ya !!!!
    // we can use just the tf from tf-idf
    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(input_file)


  if(feature equals("W")){

    var input_countvectors = tf.collect()


    /* initialising centroids */
    val sz = input_countvectors.size
    val r = scala.util.Random
    var t=0
    while( t < N){
       var n = r.nextInt(sz)
      centroids(t) = input_countvectors(n).toDense
      t = t+1
    }



    /* put all elements to some clusters first */
    var ip_copy = input_countvectors.map(x => {
      var min_dist = 1234567.toDouble
      var min_index = 0
      centroids.foreach(c => {
        var d =  euclidean_dist(x,c._2)
        if( d < min_dist){
          min_index = c._1
          min_dist = d
        }
      })
      if(!clusters_set.contains(min_index))
        clusters_set(min_index) = Set(x)
      else
        clusters_set(min_index) = clusters_set(min_index) + x

      (x,min_index)
    })




    /* N iterations */
    var itr = 1
    while(itr <= iterations){

      //calculate new centroids
      var t=0
      while(t < N){
        if(! clusters_set(t).isEmpty){
          centroids(t) = calculate_centroid_vect(clusters_set(t))
        }
        t = t+1
      }

      var ip = ip_copy.map(x=>{

        // find which cluster its closest to first
        var min_dist = 1234567.toDouble
        var min_index = 0
        var vect = x._1
        var current_cluster_index = x._2

        centroids.foreach(c => {
          var d =  euclidean_dist(vect,c._2)
          if( d < min_dist){
            min_index = c._1
            min_dist = d
          }
        })

        // now we check if there's a need to change it to different cluster
        if(current_cluster_index != min_index){   // yes it need to be changed to different cluster


          val loop = new Breaks;
          loop.breakable {

            var find_indx = 0
            while(find_indx < N ){
              // loop over
              if(find_indx!=min_index){
                if(clusters_set(find_indx).contains(vect)) //  if the set is in some other cluster remove it
                {
                  clusters_set(find_indx) = clusters_set(find_indx) - vect
                  loop.break;
                }
              }
              find_indx = find_indx + 1
            }
          }

          clusters_set(min_index) = clusters_set(min_index) + vect
          current_cluster_index = min_index
        }

        (vect,current_cluster_index)
      })

      ip_copy = ip

      itr = itr+1
    }


  }
  else if(feature equals("T")){     // using TF_IDF features

    /* IF-IDF */
  //  println("IDF feature vector !!!!!")
    val idf = new IDF().fit(tf)
    val tfidf: RDD[Vector] = idf.transform(tf)

    var input_countvectors = tfidf.collect()

    /* initialising centroids */
    val sz = input_countvectors.size
    val r = scala.util.Random
    var t=0
    //var n=150
    while( t < N){
      var n = r.nextInt(sz)
      centroids(t) = input_countvectors(n).toDense
      //n=n+150
      t = t+1
    }



    /* put all elements to some clusters first */
    var ip_copy = input_countvectors.map(x => {
      var min_dist = 1234567.toDouble
      var min_index = 0
      centroids.foreach(c => {
        var d =  euclidean_dist(x,c._2)
        if( d < min_dist){
          min_index = c._1
          min_dist = d
        }
      })
      if(!clusters_set.contains(min_index))
        clusters_set(min_index) = Set(x)
      else
        clusters_set(min_index) = clusters_set(min_index) + x

      (x,min_index)
    })

    /* N iterations */
    var itr = 1
    while(itr <= iterations){

      //calculate new centroids
      var t=0
      while(t < N){
        if(! clusters_set(t).isEmpty){
          centroids(t) = calculate_centroid_vect(clusters_set(t))
        }
        t = t+1
      }

      var ip = ip_copy.map(x=>{

        // find which cluster its closest to first
        var min_dist = 1234567.toDouble
        var min_index = 0
        var vect = x._1
        var current_cluster_index = x._2

        centroids.foreach(c => {
          var d =  euclidean_dist(vect,c._2)
          if( d < min_dist){
            min_index = c._1
            min_dist = d
          }
        })

        // now we check if there's a need to change it to different cluster
        if(current_cluster_index != min_index){   // yes it need to be changed to different cluster


          val loop = new Breaks;
          loop.breakable {

            var find_indx = 0
            while(find_indx < N ){
              // loop over
              if(find_indx!=min_index){
                if(clusters_set(find_indx).contains(vect)) //  if the set is in some other cluster remove it
                {
                  clusters_set(find_indx) = clusters_set(find_indx) - vect
                  loop.break;
                }
              }
              find_indx = find_indx + 1
            }
          }

          clusters_set(min_index) = clusters_set(min_index) + vect
          current_cluster_index = min_index
        }

        (vect,current_cluster_index)
      })

      ip_copy = ip

      itr = itr+1
    }

  }

    //top_10_freq_words in centroid vect for each cluster (N)

    //esse for each cluster

    // for each centroid calculate the top freq elements
    var top_freq_indices = HashMap.empty[Int, Set[Int]]


    centroids.foreach(c=>{

      var v= c._2
      var vect = c._2.toArray

      var top_freq = vect.sortBy(x => -x).distinct.take(10)
     // println("freq in iteration : " + c._1.toString)

      top_freq.foreach(id=>{
      // println("elem :" + id.toString + " hashcode will be found soon")   // get the index here

        val loop = new Breaks;
        loop.breakable {

            v.foreachActive((index,value)=>{

              if(value == id){

                if(!top_freq_indices.contains(c._1))
                  top_freq_indices(c._1) = Set(index)
                else
                  top_freq_indices(c._1) = top_freq_indices(c._1) + index
                loop.break;
              }

            })
        }

      })
    })


    top_freq_indices.foreach(x=>{


      var cluster_number = x._1
      var string_index_set= x._2

      string_index_set.foreach(idx =>{

        var str = get_word(input_file,hashingTF,idx)

        if(!top_freq_words.contains(cluster_number)){
          top_freq_words(cluster_number) = Set(str)
        }
        else{
          top_freq_words(cluster_number) = top_freq_words(cluster_number) + str
        }

      })

    })



    var size_cluster = HashMap.empty[Int, Int]


    clusters_set.foreach(x=>{
      if(!size_cluster.contains(x._1))
        size_cluster(x._1)= x._2.size
      else
        size_cluster(x._1)= x._2.size
    })
    //println("WSSE :  " + WSSE_calculation.toString)


    //val file = new File("Task1.json")
    val bw = new BufferedWriter(new FileWriter("Niharika_Gajam__KMeans_small_"+feature+"_5_20.json"))
    bw.write("{ \n")
    bw.write("\t \"algorithm\": \"K-Means\", \n")
    bw.write("\t \"WSSE\": "+ WSSE_calculation +", \n")
    bw.write("\t \"clusters\": [ { \n")
    for (i <- 1 to N){
      if (i!= 1){
        bw.write("\t { \n")
      }
      bw.write("\t \t \"id\":  "+ i + ", \n")
      bw.write("\t \t \"size\": " + size_cluster(i-1) + ", \n")
      bw.write("\t \t \"error\": " + esse_per_cluster(i-1) + ",\n")
      bw.write("\t \t \"terms\": [\"" + top_freq_words(i-1).toList.mkString("\",\"") + "\"]  \n")
      if (i == N) {
        bw.write("\t } ] \n")
      }
      else
        bw.write("\t }, \n")
    }
    bw.write("} \n")
    bw.close()
  }
}